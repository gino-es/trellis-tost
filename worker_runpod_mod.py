import os, json, requests, random, time, runpod
from urllib.parse import urlsplit

import numpy as np
import torch
import imageio
from typing import *
from PIL import Image
from easydict import EasyDict as edict
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.representations import Gaussian, MeshExtractResult
from trellis.utils import render_utils, postprocessing_utils

import uvicorn, uuid, asyncio
from fastapi import FastAPI, UploadFile, Form, BackgroundTasks, File, HTTPException
from fastapi.responses import JSONResponse
from enum import Enum
from collections import deque
from typing import Dict, Deque, Optional
from datetime import datetime

MAX_SEED = np.iinfo(np.int32).max

# Directories
TMP_DIR = "content"
IMG_DIR = "images"
MODEL_DIR = "models"

class TaskStatus(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"

#------------------------------------------------------------------------------------------------
# Task Manager Class
#------------------------------------------------------------------------------------------------

class TaskManager:
    def __init__(self):
        self.task_queue: Deque[str] = deque()
        self.task_status: Dict[str, dict] = {}
        self.is_processing = False
        self.processing_task: Optional[str] = None
        self.last_task_completed = datetime.now()
        # Start processing loop
        self._process_task = asyncio.create_task(self.process_queue())  

    async def process_queue(self) -> None:
        while True:  # Continuous processing loop
            try:
                if not self.is_processing and self.task_queue:
                    self.is_processing = True
                    task_id = self.task_queue.popleft()
                    self.processing_task = task_id
                    
                    idle_time = (datetime.now() - self.last_task_completed).total_seconds()
                    print(f"Starting task {task_id}. Idle time: {idle_time:.2f}s")
                    
                    task = self.task_status[task_id]
                    input_image = None
                    glb = None
                    
                    try:
                        self.update_task_status(task_id, TaskStatus.RUNNING)
                        
                        input_image = os.path.join(TMP_DIR, IMG_DIR, 
                            f"{task['image_token']}.{task['extension']}")
                        
                        state = image_to_3d(input_image)
                        glb = extract_glb(state)
                        
                        # Upload to S3 using the presigned URL
                        print(f"Uploading model to: {task['upload_url']}")
                        with open(glb, 'rb') as f:
                            response = requests.put(task['upload_url'], data=f)
                            response.raise_for_status()  # Raise exception for failed upload
                        
                        self.update_task_status(task_id, TaskStatus.SUCCESS)
                        print(f"Model uploaded successfully for task {task_id}")
                        
                    except Exception as e:
                        print(f"Error processing task {task_id}: {str(e)}")
                        self.update_task_status(task_id, TaskStatus.FAILED, error=str(e))
                    finally:
                        # Cleanup temporary files
                        for path in [input_image, glb]:
                            if path and os.path.exists(path):
                                os.remove(path)
                        
                        self.last_task_completed = datetime.now()
                        self.is_processing = False
                        self.processing_task = None
                        
                        queue_size = len(self.task_queue)
                        print(f"Completed task {task_id}. Remaining queue size: {queue_size}")
                
                if self.is_processing:
                    print(f"Processing task {self.processing_task}..., {len(self.task_queue)} tasks remaining")
                else:
                    idle_time = (datetime.now() - self.last_task_completed).total_seconds()
                    print(f"No tasks to process. Idle time: {idle_time:.2f}s")
                
                await asyncio.sleep(10)  # Check queue every 10 seconds
                
            except Exception as e:
                print(f"Error in process_queue: {e}")
                self.is_processing = False
                self.processing_task = None
                await asyncio.sleep(10)  # Wait on error before retrying

    def new_task(self, image_token: str, image_extension: str) -> None:
        print(f"Adding task {image_token} to queue. Current queue size: {len(self.task_queue)}")
        self.task_status[image_token] = {
            "status": TaskStatus.QUEUED.value,
            "image_token": image_token,
            "image_extension": image_extension,
            "upload_url": None,
            "error": None,
            "queued_at": None,
            "started_at": None,
            "completed_at": None,
            "processing_time": None,
            "queue_time": None
        }

    def update_task_status(self, task_id: str, status: TaskStatus, error: str = None) -> None:
        if task_id in self.task_status:
            now = datetime.now()
            update = {
                "status": status.value,
                "updated_at": now.isoformat()
            }
            
            if status == TaskStatus.RUNNING:
                update["started_at"] = now.isoformat()
            elif status in [TaskStatus.SUCCESS, TaskStatus.FAILED]:
                update["completed_at"] = now.isoformat()
                started_at = datetime.fromisoformat(self.task_status[task_id]["started_at"]) if self.task_status[task_id]["started_at"] else now
                queued_at = datetime.fromisoformat(self.task_status[task_id]["queued_at"])
                update["processing_time"] = (now - started_at).total_seconds()
                update["queue_time"] = (started_at - queued_at).total_seconds()
            
            if error:
                update["error"] = error
                
            self.task_status[task_id].update(update)
        else:
            print("Task not found: ", task_id)

    def queue_task(self, image_token: str, upload_url: str) -> None:

        if image_token not in self.task_queue:
            self.task_queue.append(image_token)
            self.task_status[image_token].update({"queued_at": datetime.now().isoformat(), "upload_url": upload_url})
            print(f"Task {image_token} added to queue. Current queue size: {len(self.task_queue)}")
        else:
            print(f"Task {image_token} is already in the queue.")

    def get_task_status(self, task_id: str) -> dict:
        return self.task_status.get(task_id)

# Initialize task manager
task_manager = TaskManager()
pipeline = TrellisImageTo3DPipeline.from_pretrained("/content/model")
pipeline.cuda()

def pack_state(gs: Gaussian, mesh: MeshExtractResult) -> dict:
    return {
        'gaussian': {
            **gs.init_params,
            '_xyz': gs._xyz.cpu().numpy(),
            '_features_dc': gs._features_dc.cpu().numpy(),
            '_scaling': gs._scaling.cpu().numpy(),
            '_rotation': gs._rotation.cpu().numpy(),
            '_opacity': gs._opacity.cpu().numpy(),
        },
        'mesh': {
            'vertices': mesh.vertices.cpu().numpy(),
            'faces': mesh.faces.cpu().numpy(),
        }
    }

def unpack_state(state: dict) -> Tuple[Gaussian, edict]:
    gs = Gaussian(
        aabb=state['gaussian']['aabb'],
        sh_degree=state['gaussian']['sh_degree'],
        mininum_kernel_size=state['gaussian']['mininum_kernel_size'],
        scaling_bias=state['gaussian']['scaling_bias'],
        opacity_bias=state['gaussian']['opacity_bias'],
        scaling_activation=state['gaussian']['scaling_activation'],
    )
    gs._xyz = torch.tensor(state['gaussian']['_xyz'], device='cuda')
    gs._features_dc = torch.tensor(state['gaussian']['_features_dc'], device='cuda')
    gs._scaling = torch.tensor(state['gaussian']['_scaling'], device='cuda')
    gs._rotation = torch.tensor(state['gaussian']['_rotation'], device='cuda')
    gs._opacity = torch.tensor(state['gaussian']['_opacity'], device='cuda')

    mesh = edict(
        vertices=torch.tensor(state['mesh']['vertices'], device='cuda'),
        faces=torch.tensor(state['mesh']['faces'], device='cuda'),
    )

    return gs, mesh

def image_to_3d(image_path: str) -> dict:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGBA")
    outputs = pipeline.run(
        image,
        seed=np.random.randint(0, MAX_SEED),
        formats=["gaussian", "mesh"],
        preprocess_image=True
    )

    return pack_state(outputs['gaussian'][0], outputs['mesh'][0])

def extract_glb(state: dict) -> str:
    gs, mesh = unpack_state(state)
    glb = postprocessing_utils.to_glb(
        gs, 
        mesh, 
        simplify=0.95, 
        texture_size=1024, 
        verbose=False
    )
    return glb

def download_file(url, save_dir, file_name):
    os.makedirs(save_dir, exist_ok=True)
    file_suffix = os.path.splitext(urlsplit(url).path)[1]
    file_name_with_suffix = file_name + file_suffix
    file_path = os.path.join(save_dir, file_name_with_suffix)
    response = requests.get(url)
    response.raise_for_status()
    with open(file_path, 'wb') as file:
        file.write(response.content)
    return file_path

app = FastAPI()

@app.get("/")
def default_route():
    return {"runpod worker is running..."}

@app.get("/health")
def health_check():
    return {"status": "OK"}

@app.get("/task/{task_id}")
async def get_task_status(task_id: str) -> dict:
    status = task_manager.get_task_status(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="Task not found")
    return status

@app.get("/tasks")
async def get_all_tasks() -> dict:
    return {
        "queue_length": len(task_manager.task_queue),
        "tasks": task_manager.task_status
    }

@app.post("/upload_image")
async def upload_image(file: UploadFile = File(...)):
    try:
        file_ext = os.path.splitext(file.filename)[1].lower().lstrip('.')
        if file_ext not in ['jpg', 'jpeg', 'png']:
            return JSONResponse(
                content={"error": f"Unsupported image format. Only JPG and PNG are supported: {file_ext}"},
                status_code=400
            )

        image_token = str(uuid.uuid4())
        task_manager.new_task(image_token, file_ext)
        
        save_dir = os.path.join(TMP_DIR, IMG_DIR)
        os.makedirs(save_dir, exist_ok=True)
        
        file_content = await file.read()
        file_path = os.path.join(save_dir, f"{image_token}.{file_ext}")
        
        with open(file_path, "wb") as f:
            f.write(file_content)

        return JSONResponse(content={
            "data": {
                "image_token": image_token
            }
        })
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/generate_model")
async def generate_model(request: dict):
    try:
        input_data = request["input"]
        image_token = input_data["image_token"]
        upload_url = input_data["upload_url"]

        task_manager.queue_task(image_token, upload_url)
        queue_length = len(task_manager.task_queue)
        
        return JSONResponse(content={
            "data": {
                "task_id": image_token,
                "queue_position": queue_length
            }
        })
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
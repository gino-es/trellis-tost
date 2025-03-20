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
from pydantic import BaseModel
from enum import Enum
from collections import deque
from typing import Dict, Deque, Optional
from datetime import datetime
import logging

MAX_SEED = np.iinfo(np.int32).max

# Directories
TMP_DIR = "content"
IMG_DIR = "images"
MODEL_DIR = "models"

PERSISTENT_VOLUME_PATH="/workspace"

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
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.task_status: Dict[str, dict] = {}
        self.is_processing = False
        self.processing_task: Optional[str] = None
        self.last_task_completed = datetime.now()

    async def process_queue(self) -> None:
        while True:
            try:
                # Wait for a task to be available - this is non-blocking and more efficient
                task_id = await self.task_queue.get()
                
                self.is_processing = True
                self.processing_task = task_id
                
                idle_time = (datetime.now() - self.last_task_completed).total_seconds()
                print(f"Starting task {task_id}. Idle time: {idle_time:.2f}s")
                
                await self._process_single_task(task_id)
                
                # Mark task as done
                self.task_queue.task_done()
                
            except Exception as e:
                print(f"Critical error in process_queue: {e}")
            finally:
                self.is_processing = False
                self.processing_task = None

    def new_task(self, image_token: str, image_extension: str) -> None:
        print(f"Adding task {image_token}")
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
        if image_token not in self.task_status:
            print(f"Error: Task {image_token} not found in task_status. Cannot queue.")
            return

        self.task_status[image_token].update({
            "queued_at": datetime.now().isoformat(),
            "upload_url": upload_url
        })
        # Put task in queue
        self.task_queue.put_nowait(image_token)
        print(f"Task {image_token} added to queue. Current queue size: {self.task_queue.qsize()}")

    def get_task_status(self, task_id: str) -> dict:
        return self.task_status.get(task_id)

    async def _process_single_task(self, task_id: str) -> None:
        task = self.task_status[task_id]
        input_image = None
        glb_path = None
        
        try:
            logging.info(f"Starting processing for task {task_id}")
            self.update_task_status(task_id, TaskStatus.RUNNING)
            
            input_image = os.path.join(TMP_DIR, IMG_DIR, 
                f"{task['image_token']}.{task['image_extension']}")
            
            logging.info(f"Task {task_id}: Converting image to 3D")
            state = await asyncio.get_event_loop().run_in_executor(
                None, image_to_3d, input_image
            )
            
            logging.info(f"Task {task_id}: Extracting GLB")
            glb_mesh = await asyncio.get_event_loop().run_in_executor(
                None, extract_glb, state
            )
            
            # Save the GLB to a temporary file
            save_dir = os.path.join(TMP_DIR, MODEL_DIR)
            os.makedirs(save_dir, exist_ok=True)
            glb_path = os.path.join(save_dir, f"{task_id}.glb")
            
            # Export GLB in thread pool
            await asyncio.get_event_loop().run_in_executor(
                None, glb_mesh.export, glb_path
            )
            logging.info(f"Task {task_id}: GLB saved to {glb_path}")
            
            # Upload in thread pool
            def upload_file():
                with open(glb_path, 'rb') as f:
                    response = requests.put(task['upload_url'], data=f)
                    response.raise_for_status()
                
            await asyncio.get_event_loop().run_in_executor(None, upload_file)
            
            self.update_task_status(task_id, TaskStatus.SUCCESS)
            logging.info(f"Task {task_id}: Model uploaded successfully")
            
        except Exception as e:
            logging.error(f"Error processing task {task_id}: {str(e)}", exc_info=True)
            self.update_task_status(task_id, TaskStatus.FAILED, error=str(e))
        finally:
            # Cleanup
            for path in [input_image, glb_path]:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                        logging.info(f"Task {task_id}: Cleaned up temporary file: {path}")
                    except Exception as e:
                        logging.warning(f"Task {task_id}: Failed to clean up file {path}: {str(e)}")
            
            self.last_task_completed = datetime.now()
            queue_size = self.task_queue.qsize()
            logging.info(f"Task {task_id}: Completed. Remaining queue size: {queue_size}")

# Initialize task manager
task_manager = TaskManager()

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
    
    if pipeline is None:
        raise RuntimeError("Pipeline is not initialized.")

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
    return JSONResponse(content={
        "data": status
    })

@app.get("/tasks")
async def get_all_tasks() -> dict:
    return {
        "queue_length": task_manager.task_queue.qsize(),
        "tasks": task_manager.task_status
    }

class UploadImageRequest(BaseModel):
    image_url: str
    file_extension: str

@app.post("/upload_image")
async def upload_image(request: UploadImageRequest):
    try:
        image_url = request.image_url
        file_ext = request.file_extension.lower().lstrip(".")
        logging.info(f"Received request to download image from: {image_url} with extension: {file_ext}")

        if file_ext not in ["jpg", "jpeg", "png"]:
            logging.warning(f"Rejected download of unsupported format: {file_ext}")
            return JSONResponse(
                content={"error": f"Unsupported image format. Only JPG and PNG are supported: {file_ext}"},
                status_code=400
            )

        # Generate a unique image token
        image_token = str(uuid.uuid4())
        logging.info(f"Generated token: {image_token}")
        
        save_dir = os.path.join(TMP_DIR, IMG_DIR)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f"{image_token}.{file_ext}")

        # Download the file
        response = requests.get(image_url, stream=True)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download image from S3.")

        with open(file_path, "wb") as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)

        logging.info(f"Successfully saved downloaded image: {file_path}")
        
        # Create initial task entry
        task_manager.new_task(image_token, file_ext)

        return JSONResponse(content={
            "data": {
                "image_token": image_token
            }
        })

    except Exception as e:
        logging.error(f"Error processing image download: {str(e)}", exc_info=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
class GenerateModelRequest(BaseModel):
    image_token: str
    upload_url: str

@app.post("/generate_model")
async def generate_model(request: GenerateModelRequest):
    try:
        image_token = request.image_token
        upload_url = request.upload_url

        logging.info(f"Received generate_model request for token: {image_token}")
        task_manager.queue_task(image_token, upload_url)
        queue_length = task_manager.task_queue.qsize()
        logging.info(f"Task {image_token} queued. Position in queue: {queue_length}")
        
        return JSONResponse(content={
            "data": {
                "task_id": image_token,
                "queue_position": queue_length
            }
        })
        
    except Exception as e:
        logging.error(f"Error in generate_model: {str(e)}", exc_info=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs(os.path.join(TMP_DIR, IMG_DIR), exist_ok=True)
    os.makedirs(os.path.join(TMP_DIR, MODEL_DIR), exist_ok=True)
    
    logging.info("Initializing Trellis pipeline...")
    pipeline = TrellisImageTo3DPipeline.from_pretrained(f"/{PERSISTENT_VOLUME_PATH}/model")
    pipeline.cuda()
    logging.info("Pipeline initialized successfully")
    
    # Create and set the event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Create the background task before running uvicorn
    background_task = None
    
    # Define startup and shutdown events
    @app.on_event("startup")
    async def startup_event():
        global background_task
        # Start the task manager's process_queue as a background task
        background_task = asyncio.create_task(task_manager.process_queue())
    
    @app.on_event("shutdown")
    async def shutdown_event():
        if background_task:
            background_task.cancel()
            try:
                await background_task
            except asyncio.CancelledError:
                pass
    
    # Run the app with the configured event loop
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, loop=loop)
    server = uvicorn.Server(config)
    loop.run_until_complete(server.serve())
import os
import json
import uuid
import requests
import random
import time
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
from fastapi import FastAPI, UploadFile, Form, BackgroundTasks
from pydantic import BaseModel
import uvicorn

app = FastAPI()

MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = "/content"

def preprocess_image(image_path: str) -> Tuple[str, Image.Image]:
    trial_id = f"{int(time.time() * 1000)}-{uuid.uuid4()}"
    image = Image.open(image_path).convert("RGBA")
    processed_image = pipeline.preprocess_image(image)
    processed_image.save(f"{TMP_DIR}/{trial_id}.png")
    return trial_id, processed_image

def pack_state(gs: Gaussian, mesh: MeshExtractResult, trial_id: str) -> dict:
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
        },
        'trial_id': trial_id,
    }

def unpack_state(state: dict) -> Tuple[Gaussian, edict, str]:
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

    return gs, mesh, state['trial_id']

def image_to_3d(image_path: str, seed: int = 0, randomize_seed: bool = True,
                ss_guidance_strength: float = 7.5, ss_sampling_steps: int = 12,
                slat_guidance_strength: float = 3.0, slat_sampling_steps: int = 12) -> Tuple[dict, str]:
    trial_id, _ = preprocess_image(image_path)
    if randomize_seed:
        seed = np.random.randint(0, MAX_SEED)

    outputs = pipeline.run(
        Image.open(f"{TMP_DIR}/{trial_id}.png"),
        seed=seed,
        formats=["gaussian", "mesh"],
        preprocess_image=False,
        sparse_structure_sampler_params={
            "steps": ss_sampling_steps,
            "cfg_strength": ss_guidance_strength,
        },
        slat_sampler_params={
            "steps": slat_sampling_steps,
            "cfg_strength": slat_guidance_strength,
        },
    )

    video = render_utils.render_video(outputs['gaussian'][0], num_frames=120)['color']
    video_geo = render_utils.render_video(outputs['mesh'][0], num_frames=120)['normal']
    video = [np.concatenate([video[i], video_geo[i]], axis=1) for i in range(len(video))]
    video_path = f"{TMP_DIR}/{trial_id}.mp4"
    imageio.mimsave(video_path, video, fps=15)

    state = pack_state(outputs['gaussian'][0], outputs['mesh'][0], str(trial_id))
    return state, video_path

def extract_glb(state: dict, mesh_simplify: float = 0.95, texture_size: int = 1024) -> str:
    gs, mesh, trial_id = unpack_state(state)
    glb = postprocessing_utils.to_glb(gs, mesh, simplify=mesh_simplify, texture_size=texture_size, verbose=False)
    glb_path = f"{TMP_DIR}/{trial_id}.glb"
    glb.export(glb_path)
    return glb_path

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

pipeline = TrellisImageTo3DPipeline.from_pretrained("/content/model")
pipeline.cuda()

def generate(input):
    values = input["input"]

    input_image = values['input_image']
    input_image = download_file(url=input_image, save_dir=TMP_DIR, file_name='input_image')
    seed = values['seed']
    randomize_seed = values['randomize_seed']
    ss_guidance_strength = values['ss_guidance_strength']
    ss_sampling_steps = values['ss_sampling_steps']
    slat_guidance_strength = values['slat_guidance_strength']
    slat_sampling_steps = values['slat_sampling_steps']
    mesh_simplify = values['mesh_simplify']
    texture_size = values['texture_size']

    state, video_path = image_to_3d(image_path=input_image, 
                                    seed=seed, 
                                    randomize_seed=randomize_seed, 
                                    ss_guidance_strength=ss_guidance_strength, 
                                    ss_sampling_steps=ss_sampling_steps,
                                    slat_guidance_strength=slat_guidance_strength,
                                    slat_sampling_steps=slat_sampling_steps)
    glb_path = extract_glb(state=state, mesh_simplify=mesh_simplify, texture_size=texture_size)

    result = [video_path, [glb_path, input_image]]
    try:
        notify_uri = values['notify_uri']
        del values['notify_uri']
        notify_token = values['notify_token']
        del values['notify_token']
        job_id = values['job_id']
        del values['job_id']
        default_filename = os.path.basename(result[0])
        with open(result[0], "rb") as file:
            files = {default_filename: file.read()}
        for path in result[1]:
            filename = os.path.basename(path)
            with open(path, "rb") as file:
                files[filename] = file.read()
        notify_payload = {"jobId": job_id, "result": str(result), "status": "DONE"}
        web_notify_uri = os.getenv('com_camenduru_web_notify_uri')
        web_notify_token = os.getenv('com_camenduru_web_notify_token')
        if(notify_uri == "notify_uri"):
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
        else:
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            requests.post(notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        return {"jobId": job_id, "result": str(result), "status": "DONE"}
    except Exception as e:
        error_payload = {"jobId": job_id, "status": "FAILED"}
        try:
            if(notify_uri == "notify_uri"):
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            else:
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
                requests.post(notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        except:
            pass
        return {"jobId": job_id, "result": f"FAILED: {str(e)}", "status": "FAILED"}
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(glb_path):
            os.remove(glb_path)
        if os.path.exists(input_image):
            os.remove(input_image)

@app.get("/")
def default_route():
    return {"runpod worker is running..."}

@app.get("/health")
def health_check():
    return {"status": "OK"}

@app.post("/generate")
def generate_route(input: dict, background_tasks: BackgroundTasks):
    background_tasks.add_task(generate, input)
        
uvicorn.run(app, host="0.0.0.0", port=8000)
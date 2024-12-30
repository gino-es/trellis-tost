#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

echo "Updating package lists and installing system dependencies..."
sudo apt update -y && sudo apt install -y \
    build-essential libgl1 libglib2.0-0 zlib1g-dev \
    libncurses5-dev libgdbm-dev libnss3-dev libssl-dev \
    libreadline-dev libffi-dev nano aria2 curl unzip unrar ffmpeg git-lfs && \
    sudo apt clean

echo "Installing NVIDIA CUDA Toolkit..."
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    https://developer.download.nvidia.com/compute/cuda/12.6.2/local_installers/cuda_12.6.2_560.35.03_linux.run -d /content -o cuda_12.6.2.run && \
    sh /content/cuda_12.6.2.run --silent --toolkit && \
    echo "/usr/local/cuda/lib64" | sudo tee -a /etc/ld.so.conf && sudo ldconfig && \
    rm -f /content/cuda_12.6.2.run

echo "Installing Python dependencies..."
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 \
    torchaudio==2.5.1+cu124 torchtext==0.18.0 torchdata==0.8.0 \
    --extra-index-url https://download.pytorch.org/whl/cu124 && \
    pip install xformers==0.0.28.post3 && \
    pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl && \
    pip install opencv-contrib-python imageio imageio-ffmpeg ffmpeg-python av runpod easydict rembg onnxruntime \
    onnxruntime-gpu numpy==2.0.0 plyfile huggingface-hub safetensors

echo "Installing TRELLIS-specific dependencies..."
pip install git+https://github.com/NVlabs/nvdiffrast trimesh xatlas pyvista pymeshfix igraph spconv-cu120 && \
    pip install https://github.com/camenduru/wheels/releases/download/3090/kaolin-0.17.0-cp310-cp310-linux_x86_64.whl && \
    pip install https://github.com/camenduru/wheels/releases/download/3090/diso-0.1.4-cp310-cp310-linux_x86_64.whl && \
    pip install https://github.com/camenduru/wheels/releases/download/3090/utils3d-0.0.2-py3-none-any.whl && \
    pip install https://huggingface.co/spaces/JeffreyXiang/TRELLIS/resolve/main/wheels/nvdiffrast-0.3.3-cp310-cp310-linux_x86_64.whl && \
    pip install https://huggingface.co/spaces/JeffreyXiang/TRELLIS/resolve/main/wheels/diff_gaussian_rasterization-0.0.0-cp310-cp310-linux_x86_64.whl

echo "Downloading model files..."
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/JeffreyXiang/TRELLIS-image-large/raw/main/pipeline.json -d /content/model -o pipeline.json && \
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/JeffreyXiang/TRELLIS-image-large/raw/main/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16.json -d /content/model/ckpts -o slat_dec_gs_swin8_B_64l8gs32_fp16.json && \
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/JeffreyXiang/TRELLIS-image-large/resolve/main/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16.safetensors -d /content/model/ckpts -o slat_dec_gs_swin8_B_64l8gs32_fp16.safetensors && \
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/JeffreyXiang/TRELLIS-image-large/raw/main/ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16.json -d /content/model/ckpts -o slat_dec_mesh_swin8_B_64l8m256c_fp16.json && \
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/JeffreyXiang/TRELLIS-image-large/resolve/main/ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16.safetensors -d /content/model/ckpts -o slat_dec_mesh_swin8_B_64l8m256c_fp16.safetensors && \
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/JeffreyXiang/TRELLIS-image-large/raw/main/ckpts/slat_dec_rf_swin8_B_64l8r16_fp16.json -d /content/model/ckpts -o slat_dec_rf_swin8_B_64l8r16_fp16.json && \
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/JeffreyXiang/TRELLIS-image-large/resolve/main/ckpts/slat_dec_rf_swin8_B_64l8r16_fp16.safetensors -d /content/model/ckpts -o slat_dec_rf_swin8_B_64l8r16_fp16.safetensors && \
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/JeffreyXiang/TRELLIS-image-large/raw/main/ckpts/slat_enc_swin8_B_64l8_fp16.json -d /content/model/ckpts -o slat_enc_swin8_B_64l8_fp16.json && \
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/JeffreyXiang/TRELLIS-image-large/resolve/main/ckpts/slat_enc_swin8_B_64l8_fp16.safetensors -d /content/model/ckpts -o slat_enc_swin8_B_64l8_fp16.safetensors && \
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/JeffreyXiang/TRELLIS-image-large/raw/main/ckpts/slat_flow_img_dit_L_64l8p2_fp16.json -d /content/model/ckpts -o slat_flow_img_dit_L_64l8p2_fp16.json && \
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/JeffreyXiang/TRELLIS-image-large/resolve/main/ckpts/slat_flow_img_dit_L_64l8p2_fp16.safetensors -d /content/model/ckpts -o slat_flow_img_dit_L_64l8p2_fp16.safetensors && \
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/JeffreyXiang/TRELLIS-image-large/raw/main/ckpts/ss_dec_conv3d_16l8_fp16.json -d /content/model/ckpts -o ss_dec_conv3d_16l8_fp16.json && \
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/JeffreyXiang/TRELLIS-image-large/resolve/main/ckpts/ss_dec_conv3d_16l8_fp16.safetensors -d /content/model/ckpts -o ss_dec_conv3d_16l8_fp16.safetensors && \
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/JeffreyXiang/TRELLIS-image-large/raw/main/ckpts/ss_enc_conv3d_16l8_fp16.json -d /content/model/ckpts -o ss_enc_conv3d_16l8_fp16.json && \
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/JeffreyXiang/TRELLIS-image-large/resolve/main/ckpts/ss_enc_conv3d_16l8_fp16.safetensors -d /content/model/ckpts -o ss_enc_conv3d_16l8_fp16.safetensors && \
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/JeffreyXiang/TRELLIS-image-large/raw/main/ckpts/ss_flow_img_dit_L_16l8_fp16.json -d /content/model/ckpts -o ss_flow_img_dit_L_16l8_fp16.json && \
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/JeffreyXiang/TRELLIS-image-large/resolve/main/ckpts/ss_flow_img_dit_L_16l8_fp16.safetensors -d /content/model/ckpts -o ss_flow_img_dit_L_16l8_fp16.safetensors && \
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://github.com/facebookresearch/dinov2/zipball/main -d /home/camenduru/.cache/torch/hub -o main.zip && \
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth -d /home/camenduru/.cache/torch/hub/checkpoints -o dinov2_vitl14_reg4_pretrain.pth && \
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx -d /home/camenduru/.u2net -o u2net.onnx && \
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://github.com/facebookresearch/dinov2/zipball/main -d /home/camenduru/.cache/torch/hub -o main.zip

echo "Environment setup complete. Starting worker..."
cd /content/TRELLIS
python worker_runpod_mod.py

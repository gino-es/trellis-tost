#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

# Define the persistent volume path
PERSISTENT_VOLUME_PATH="/workspace"
CUDA_INSTALL_PATH="$PERSISTENT_VOLUME_PATH/cuda"
TORCH_CACHE_PATH="$PERSISTENT_VOLUME_PATH/.cache/torch/hub"

# Set TORCH_HOME to use the persistent cache directory
export TORCH_HOME="$PERSISTENT_VOLUME_PATH/.cache/torch"

# Create the directory if it doesn't exist
mkdir -p $PERSISTENT_VOLUME_PATH $TORCH_CACHE_PATH

# Ensure the script has the necessary permissions
chmod -R 777 $PERSISTENT_VOLUME_PATH

# Install CUDA if not already installed
if [ ! -d "$CUDA_INSTALL_PATH" ]; then
    echo "Installing NVIDIA CUDA Toolkit..."
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
        https://developer.download.nvidia.com/compute/cuda/12.6.2/local_installers/cuda_12.6.2_560.35.03_linux.run -d $PERSISTENT_VOLUME_PATH -o cuda_12.6.2.run && \
        sh $PERSISTENT_VOLUME_PATH/cuda_12.6.2.run --silent --toolkit --installpath=$CUDA_INSTALL_PATH && \
        echo "$CUDA_INSTALL_PATH/lib64" | sudo tee -a /etc/ld.so.conf && sudo ldconfig && \
        rm -f $PERSISTENT_VOLUME_PATH/cuda_12.6.2.run
else
    echo "CUDA Toolkit already installed."
fi

# Update PATH and LD_LIBRARY_PATH
export PATH="$CUDA_INSTALL_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_INSTALL_PATH/lib64:$LD_LIBRARY_PATH"

# Check if model files are already downloaded
if [ ! -f $PERSISTENT_VOLUME_PATH/model/pipeline.json ]; then
    echo "Downloading model files..."
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/JeffreyXiang/TRELLIS-image-large/raw/main/pipeline.json -d $PERSISTENT_VOLUME_PATH/model -o pipeline.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/JeffreyXiang/TRELLIS-image-large/raw/main/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16.json -d $PERSISTENT_VOLUME_PATH/model/ckpts -o slat_dec_gs_swin8_B_64l8gs32_fp16.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/JeffreyXiang/TRELLIS-image-large/resolve/main/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16.safetensors -d $PERSISTENT_VOLUME_PATH/model/ckpts -o slat_dec_gs_swin8_B_64l8gs32_fp16.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/JeffreyXiang/TRELLIS-image-large/raw/main/ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16.json -d $PERSISTENT_VOLUME_PATH/model/ckpts -o slat_dec_mesh_swin8_B_64l8m256c_fp16.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/JeffreyXiang/TRELLIS-image-large/resolve/main/ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16.safetensors -d $PERSISTENT_VOLUME_PATH/model/ckpts -o slat_dec_mesh_swin8_B_64l8m256c_fp16.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/JeffreyXiang/TRELLIS-image-large/raw/main/ckpts/slat_dec_rf_swin8_B_64l8r16_fp16.json -d $PERSISTENT_VOLUME_PATH/model/ckpts -o slat_dec_rf_swin8_B_64l8r16_fp16.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/JeffreyXiang/TRELLIS-image-large/resolve/main/ckpts/slat_dec_rf_swin8_B_64l8r16_fp16.safetensors -d $PERSISTENT_VOLUME_PATH/model/ckpts -o slat_dec_rf_swin8_B_64l8r16_fp16.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/JeffreyXiang/TRELLIS-image-large/raw/main/ckpts/slat_enc_swin8_B_64l8_fp16.json -d $PERSISTENT_VOLUME_PATH/model/ckpts -o slat_enc_swin8_B_64l8_fp16.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/JeffreyXiang/TRELLIS-image-large/resolve/main/ckpts/slat_enc_swin8_B_64l8_fp16.safetensors -d $PERSISTENT_VOLUME_PATH/model/ckpts -o slat_enc_swin8_B_64l8_fp16.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/JeffreyXiang/TRELLIS-image-large/raw/main/ckpts/slat_flow_img_dit_L_64l8p2_fp16.json -d $PERSISTENT_VOLUME_PATH/model/ckpts -o slat_flow_img_dit_L_64l8p2_fp16.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/JeffreyXiang/TRELLIS-image-large/resolve/main/ckpts/slat_flow_img_dit_L_64l8p2_fp16.safetensors -d $PERSISTENT_VOLUME_PATH/model/ckpts -o slat_flow_img_dit_L_64l8p2_fp16.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/JeffreyXiang/TRELLIS-image-large/raw/main/ckpts/ss_dec_conv3d_16l8_fp16.json -d $PERSISTENT_VOLUME_PATH/model/ckpts -o ss_dec_conv3d_16l8_fp16.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/JeffreyXiang/TRELLIS-image-large/resolve/main/ckpts/ss_dec_conv3d_16l8_fp16.safetensors -d $PERSISTENT_VOLUME_PATH/model/ckpts -o ss_dec_conv3d_16l8_fp16.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/JeffreyXiang/TRELLIS-image-large/raw/main/ckpts/ss_enc_conv3d_16l8_fp16.json -d $PERSISTENT_VOLUME_PATH/model/ckpts -o ss_enc_conv3d_16l8_fp16.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/JeffreyXiang/TRELLIS-image-large/resolve/main/ckpts/ss_enc_conv3d_16l8_fp16.safetensors -d $PERSISTENT_VOLUME_PATH/model/ckpts -o ss_enc_conv3d_16l8_fp16.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/JeffreyXiang/TRELLIS-image-large/raw/main/ckpts/ss_flow_img_dit_L_16l8_fp16.json -d $PERSISTENT_VOLUME_PATH/model/ckpts -o ss_flow_img_dit_L_16l8_fp16.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/JeffreyXiang/TRELLIS-image-large/resolve/main/ckpts/ss_flow_img_dit_L_16l8_fp16.safetensors -d $PERSISTENT_VOLUME_PATH/model/ckpts -o ss_flow_img_dit_L_16l8_fp16.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://github.com/facebookresearch/dinov2/zipball/main -d $TORCH_CACHE_PATH -o main.zip && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth -d $TORCH_CACHE_PATH/checkpoints -o dinov2_vitl14_reg4_pretrain.pth && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx -d /home/camenduru/.u2net -o u2net.onnx
else
    echo "Model files already downloaded."
fi

echo "Environment setup complete. Starting worker..."
cd /content/TRELLIS
python worker_runpod_mod.py

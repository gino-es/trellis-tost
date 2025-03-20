FROM ubuntu:22.04

WORKDIR /content

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=True
ENV PATH="/home/camenduru/.local/bin:${PATH}"

# Install essential system tools and packages
RUN apt update -y && apt install -y software-properties-common sudo wget git && \
    add-apt-repository -y ppa:git-core/ppa && apt update -y && \
    apt install -y python-is-python3 python3-pip \
    build-essential libgl1 libglib2.0-0 zlib1g-dev \
    libncurses5-dev libgdbm-dev libnss3-dev libssl-dev \
    libreadline-dev libffi-dev nano aria2 curl unzip unrar ffmpeg git-lfs && \
    apt clean

# Install Python dependencies
RUN pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 \
    torchaudio==2.5.1+cu124 torchtext==0.18.0 torchdata==0.8.0 \
    --extra-index-url https://download.pytorch.org/whl/cu124 && \
    pip install xformers==0.0.28.post3 && \
    pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl && \
    pip install opencv-contrib-python imageio imageio-ffmpeg ffmpeg-python av runpod easydict rembg onnxruntime \
    onnxruntime-gpu numpy==2.0.0 plyfile huggingface-hub safetensors

# Install TRELLIS-specific dependencies
RUN pip install git+https://github.com/NVlabs/nvdiffrast trimesh xatlas pyvista pymeshfix igraph spconv-cu120 && \
    pip install https://github.com/camenduru/wheels/releases/download/3090/kaolin-0.17.0-cp310-cp310-linux_x86_64.whl && \
    pip install https://github.com/camenduru/wheels/releases/download/3090/diso-0.1.4-cp310-cp310-linux_x86_64.whl && \
    pip install https://github.com/camenduru/wheels/releases/download/3090/utils3d-0.0.2-py3-none-any.whl && \
    pip install https://huggingface.co/spaces/JeffreyXiang/TRELLIS/resolve/main/wheels/nvdiffrast-0.3.3-cp310-cp310-linux_x86_64.whl && \
    pip install https://huggingface.co/spaces/JeffreyXiang/TRELLIS/resolve/main/wheels/diff_gaussian_rasterization-0.0.0-cp310-cp310-linux_x86_64.whl

# Add user and set permissions
RUN adduser --disabled-password --gecos '' camenduru && \
    adduser camenduru sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R camenduru:camenduru /content && \
    chmod -R 777 /content

# Clone TRELLIS repository
RUN git clone --recursive https://github.com/Microsoft/TRELLIS /content/TRELLIS

# Copy the custom script
COPY ./worker_runpod_mod.py /content/TRELLIS/worker_runpod_mod.py
COPY --chmod=0755 ./run.sh /content/run.sh

# Run the main script and keep the container alive
CMD ["/bin/bash", "-c", "/content/run.sh && tail -f /dev/null"]

# CMD ["sleep", "infinity"]

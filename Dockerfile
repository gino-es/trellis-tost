FROM ubuntu:22.04

WORKDIR /content

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=True
ENV PATH="/home/camenduru/.local/bin:/usr/local/cuda/bin:${PATH}"

# Install only essential system tools and packages to reduce the image size
RUN apt update -y && apt install -y software-properties-common sudo wget git && \
    add-apt-repository -y ppa:git-core/ppa && apt update -y && \
    apt install -y python-is-python3 python3-pip && \
    adduser --disabled-password --gecos '' camenduru && \
    adduser camenduru sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R camenduru:camenduru /content && \
    chmod -R 777 /content

RUN git clone --recursive https://github.com/Microsoft/TRELLIS /content/TRELLIS

# Copy the custom script
COPY ./worker_runpod_mod.py /content/TRELLIS/worker_runpod_mod.py
COPY --chmod=0755 ./run.sh /content/run.sh

# Run the script at container runtime
CMD ["/bin/bash", "/content/run.sh"]

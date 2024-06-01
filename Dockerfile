FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04


ENV TZ=Europe/Amsterdam
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install python3.8
RUN : \
    && apt-get update \
    && apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository -y ppa:deadsnakes \
    && apt-get install -y --no-install-recommends python3.10-venv \
    && apt-get install libpython3.10-dev -y \
    && apt-get clean \
    && :
    
# Add env to PATH
RUN python3.10 -m venv /venv
ENV PATH=/venv/bin:$PATH

# Install ASAP
RUN : \
    && apt-get update \
    && apt-get -y install curl \
    && curl --remote-name --location "https://github.com/computationalpathologygroup/ASAP/releases/download/ASAP-2.2-(Nightly)/ASAP-2.2-Ubuntu2204.deb" \
    && dpkg --install ASAP-2.2-Ubuntu2204.deb || true \
    && apt-get -f install --fix-missing --fix-broken --assume-yes \
    && ldconfig -v \
    && apt-get clean \
    && echo "/opt/ASAP/bin" > /venv/lib/python3.10/site-packages/asap.pth \
    && rm ASAP-2.2-Ubuntu2204.deb \
    && :

# Install algorithm
###### not sure whether 'pip install torch torchvision torchaudio' is necessary
############

###########
COPY ./ /home/user/pathology-tiger-algorithm/
COPY nnUnet_config/ /home/user/nnUnet_config/
RUN : \
    && pip install wheel==0.43.0 \
    #&& pip install /home/user/pathology-tiger-algorithm/nnUnet_config \
    && pip install /home/user/pathology-tiger-algorithm \
    # && pip install torch torchvision torchaudio \
    && pip install nnunetv2==2.4.2 \
    && pip install scikit-image==0.23.2 \
    && rm -r /home/user/pathology-tiger-algorithm \
    # && rm -r /home/user/pathology-tiger-algorithm \
    && :

##
#COPY ./ /home/user/pathology-tiger-algorithm/nnUnet_config/
# Copy only the nnUnet_config folder into the specified directory in the Docker image
COPY nnUnet_config/ /home/user/nnUnet_config/

##

# Make user
RUN groupadd -r user && useradd -r -g user user
RUN chown user /home/user/
RUN mkdir /output/
RUN chown user /output/
USER user
WORKDIR /home/user

# Cmd and entrypoint
CMD ["-mtiger_nnunet_v2"]
ENTRYPOINT ["python"]
#ENTRYPOINT ["/bin/bash"]

# Compute requirements
LABEL processor.cpus="8"
LABEL processor.cpu.capabilities="null"
LABEL processor.memory="31G"
LABEL processor.gpu_count="1"
LABEL processor.gpu.compute_capability="null"
LABEL processor.gpu.memory="15G"
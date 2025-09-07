# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Use NVIDIA PyTorch container as base image
FROM nvcr.io/nvidia/pytorch:24.10-py3

# Install basic tools
RUN apt-get update && apt-get install -y git tree ffmpeg wget && \
    rm /bin/sh && ln -s /bin/bash /bin/sh && ln -s /lib64/libcuda.so.1 /lib64/libcuda.so

# Copy the cosmos-predict1.yaml and requirements.txt files to the container
# cosmos predict yaml installs cuda 12.4 base image is on cuda 12.6
COPY ./cosmos-predict1.yaml /cosmos-predict1.yaml
COPY ./requirements.txt /requirements.txt

ENV CONDA_DIR=/opt/conda
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH=${CONDA_DIR}/bin:${PATH}
ENV ENVNAME=cosmos-predict1
ENV ENVDIR=${CONDA_DIR}/envs/${ENVNAME}
ENV CUDA_HOME=${ENVDIR}
ENV CONDA_ACCEPT_TERMS=true
ENV PATH=${ENVDIR}/bin:${PATH}

RUN echo "Installing dependencies. This will take a while..." && \
    rm -rf ${CONDA_DIR} /tmp/mamba.sh && \
    mkdir -p ${CONDA_DIR} && \
    wget --no-hsts --quiet "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" -O /tmp/mamba.sh && \
    /bin/bash /tmp/mamba.sh -b -u -p ${CONDA_DIR} && \
    rm /tmp/mamba.sh && \
    conda clean --tarballs --index-cache --packages --yes && \
    find ${CONDA_DIR} -follow -type f -name '*.a' -delete && \
    find ${CONDA_DIR} -follow -type f -name '*.pyc' -delete && \
    conda clean --force-pkgs-dirs --all --yes 

# use bash for subsequent RUN, no need to do RUN /bin/bash -c 
SHELL ["/bin/bash", "-c"]

RUN mamba env create --file /${ENVNAME}.yaml && \
    source ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate ${ENVNAME} && \
    pip install --no-cache-dir -r /requirements.txt && \
    ln -sf ${ENVDIR}/lib/python3.10/site-packages/nvidia/*/include/* ${ENVDIR}/include/ && \
    ln -sf ${ENVDIR}/lib/python3.10/site-packages/nvidia/*/include/* ${ENVDIR}/include/python3.10 && \
    ln -sf ${ENVDIR}/lib/python3.10/site-packages/triton/backends/nvidia/include/* ${ENVDIR}/include/ && \
    pip install transformer-engine[pytorch]==1.12.0 && \
    git clone https://github.com/NVIDIA/apex && \
    ln -sf ${ENVDIR}/lib/python3.10/site-packages/triton/backends/nvidia/include/crt ${ENVDIR}/include/ && \
    pip install git+https://github.com/NVlabs/nvdiffrast.git && \
    echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate ${ENVNAME}" >> /etc/skel/.bashrc && \
    echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate ${ENVNAME}" >> ~/.bashrc 

ENV PATH=${ENVDIR}/bin:${PATH}

CMD ["/bin/bash"]

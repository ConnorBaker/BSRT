ARG CUDA_VERSION=11.8.0
FROM mambaorg/micromamba:1.0-jammy-cuda-${CUDA_VERSION}

RUN \
    --mount=type=cache,mode=0755,target=/opt/conda/pkgs \
    micromamba install \
        --yes \
        --name base \
        --channel conda-forge \
        git==2.38.1 \
        pip==22.3.1 \
        python==3.10.6 \
    && micromamba clean \
        --all \
        --yes

COPY --chown=$MAMBA_USER:$MAMBA_USER . /BSRT
WORKDIR /BSRT

ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN \
    --mount=type=cache,mode=0755,target=/home/$MAMBA_USER/.cache/pip \
    pip install \
        --pre \
        --extra-index-url https://download.pytorch.org/whl/nightly/cu117 \
        .[tune]

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN \
    --mount=type=cache,target=/var/cache/micromamba \
    micromamba install \
        --yes \
        --name base \
        --channel conda-forge \
        python==3.10.6 \
        pip==22.3.1 \
        git==2.38.1 \
        fastai::opencv-python-headless==4.6.0.66 \
    && micromamba clean \
        --all \
        --yes

COPY --chown=$MAMBA_USER:$MAMBA_USER . /BSRT
WORKDIR /BSRT

ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN \
    --mount=type=cache,target=/var/cache/pip \
    pip install \
        --no-cache-dir \
        --editable .

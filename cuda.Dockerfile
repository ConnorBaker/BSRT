FROM mambaorg/micromamba:0.27.0-focal-cuda-11.6.2
WORKDIR /gridai/project
COPY --chown=$MAMBA_USER:$MAMBA_USER . .
ARG TMPDIR=/var/tmp
RUN micromamba install -y -n base -f ./conda/environment.yml && \
    micromamba clean --all --yes

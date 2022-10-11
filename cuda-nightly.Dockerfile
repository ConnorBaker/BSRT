FROM mambaorg/micromamba:0.27.0-jammy-cuda-11.7.1
WORKDIR /gridai/project
COPY --chown=$MAMBA_USER:$MAMBA_USER . .
ARG TMPDIR=/var/tmp
RUN micromamba install -y -n base -f ./conda/environment-nightly.yml && \
    micromamba clean --all --yes

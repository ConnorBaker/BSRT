FROM mambaorg/micromamba:0.27.0-jammy
WORKDIR /gridai/project
COPY --chown=$MAMBA_USER:$MAMBA_USER . .
ARG TMPDIR=/var/tmp
RUN micromamba install -y -n base -f ./conda/environment-mac.yml && \
    micromamba clean --all --yes

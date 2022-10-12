FROM mambaorg/micromamba:0.27.0-focal-cuda-11.6.2
COPY --chown=$MAMBA_USER:$MAMBA_USER ./conda/environment.yml /tmp/env.yml
RUN micromamba install -y -n base -f /tmp/env.yml && \
    micromamba clean --all --yes
ENV PATH "$MAMBA_ROOT_PREFIX/bin:$PATH"

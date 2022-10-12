FROM mambaorg/micromamba:0.27.0-jammy-cuda-11.7.1
COPY --chown=$MAMBA_USER:$MAMBA_USER ./conda/environment-nightly.yml /tmp/env.yml
RUN micromamba install -y -n base -f /tmp/env.yml && \
    micromamba clean --all --yes
ENV PATH "$MAMBA_ROOT_PREFIX/bin:$PATH"

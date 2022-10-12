FROM mambaorg/micromamba:0.27.0-jammy
COPY --chown=$MAMBA_USER:$MAMBA_USER ./conda/environment-mac.yml /tmp/env.yml
RUN micromamba install -y -n base -f /tmp/env.yml && \
    micromamba clean --all --yes
ENV PATH "$MAMBA_ROOT_PREFIX/bin:$PATH"

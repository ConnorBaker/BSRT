ARG BASE_IMAGE
FROM ${BASE_IMAGE}

COPY --chown=$MAMBA_USER:$MAMBA_USER . /BSRT
ARG CONDA_ENV_FILE
RUN micromamba install -y -n base -f /BSRT/${CONDA_ENV_FILE} && \
    micromamba clean --all --yes
ENV PATH $MAMBA_ROOT_PREFIX/bin:$PATH

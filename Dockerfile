ARG BASE_IMAGE
FROM ${BASE_IMAGE}

COPY . /BSRT
WORKDIR /BSRT
RUN pip --no-cache-dir install pip==22.3 --upgrade && \
    pip --no-cache-dir install -e .

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

COPY . /BSRT
WORKDIR /BSRT
RUN pip install pip==22.3 --upgrade && pip install -e .

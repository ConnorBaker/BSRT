ARG BASE_IMAGE
FROM ${BASE_IMAGE}

COPY . /BSRT
WORKDIR /BSRT
RUN pip install -v -r requirements.txt

#!/usr/bin/env bash

set -eou pipefail

CONDA_ENV_FILE="${CONDA_ENV_FILE:-conda/environment-nightly.yml}"
MICROMAMBA_VERSION="${MICROMAMBA_VERSION:-0.27.0}"
OS_NAME="${OS_NAME:-jammy}"
WITH_CUDA="${WITH_CUDA:-true}"
CUDA_VERSION="${CUDA_VERSION:-11.7.1}"

if [[ "${WITH_CUDA}" == "true" ]]; then
    BASE_IMAGE="mambaorg/micromamba:${MICROMAMBA_VERSION}-${OS_NAME}-cuda-${CUDA_VERSION}"
    TAG="connorbaker01/bsrt-${OS_NAME}-cuda-${CUDA_VERSION}"
else
    BASE_IMAGE="mambaorg/micromamba:${MICROMAMBA_VERSION}-${OS_NAME}"
    TAG="connorbaker01/bsrt-${OS_NAME}"
fi

echo "---------------------------------------------"
echo "Build Arguments:"
echo "CONDA_ENV_FILE: ${CONDA_ENV_FILE}"
echo "MICROMAMBA_VERSION: ${MICROMAMBA_VERSION}"
echo "OS_NAME: ${OS_NAME}"
echo "WITH_CUDA: ${WITH_CUDA}"
if [[ "${WITH_CUDA}" == "true" ]]; then
    echo "CUDA_VERSION: ${CUDA_VERSION}"
fi
echo "BASE_IMAGE: ${BASE_IMAGE}"
echo "TAG: ${TAG}"
echo "---------------------------------------------"

BASE_IMAGE="${BASE_IMAGE}" \
CONDA_ENV_FILE="${CONDA_ENV_FILE}" \
docker buildx build --platform linux/amd64 \
    -t "${TAG}" \
    --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
    --build-arg "CONDA_ENV_FILE=${CONDA_ENV_FILE}" \
    --file ./Dockerfile \
    --push .

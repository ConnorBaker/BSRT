#!/usr/bin/env bash

set -euxo pipefail

CUDA_VERSION="${CUDA_VERSION:-11.8.0}"
BASE_IMAGE="mambaorg/micromamba:1.0-jammy-cuda-${CUDA_VERSION}"
TAG="connorbaker01/bsrt:jammy-cuda-${CUDA_VERSION}"

echo "---------------------------------------------"
echo "Build Arguments:"
echo "CUDA_VERSION: ${CUDA_VERSION}"
echo "BASE_IMAGE: ${BASE_IMAGE}"
echo "TAG: ${TAG}"
echo "---------------------------------------------"

BASE_IMAGE="${BASE_IMAGE}" \
docker buildx build \
    --progress=plain \
    --platform linux/amd64 \
    -t "${TAG}" \
    --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
    --file ./Dockerfile .

echo "Build Complete"
echo "Testing Image..."

docker run --platform linux/amd64 --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm "${TAG}" python -c "import torch; print(f'torch.__version__: {torch.__version__}'); print(f'torch.cuda.is_available(): {torch.cuda.is_available()}')"

echo "Test Complete"

echo "To push the image, run: docker push ${TAG}"

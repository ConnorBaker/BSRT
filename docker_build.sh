#!/usr/bin/env bash

set -eou pipefail

RELEASE="${RELEASE:-22.09}"
BASE_IMAGE="nvcr.io/nvidia/pytorch:${RELEASE}-py3"
TAG="connorbaker01/bsrt:${RELEASE}"

echo "---------------------------------------------"
echo "Build Arguments:"
echo "RELEASE: ${RELEASE}"
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

#!/usr/bin/env bash

set -euo pipefail

BUILDKIT_ORG="moby"
BUILDKIT_REPO="buildkit"
BUILDKIT_PREFIX="$BUILDKIT_ORG/$BUILDKIT_REPO"

# BUILDKIT_TAG="v0.11.2"
# BUILDKIT_SHA256="f433543742bd53e18f9791261a8b4cf66463c203cb40f4d47f1bbf86f48ccb45"
BUILDKIT_TAG="master"
BUILDKIT_SHA256="f68bc91b93320a197f6587f124766392c0366bd766b22afed99a4e8c817f2328"
BUILDKIT_SUFFIX="$BUILDKIT_TAG@sha256:$BUILDKIT_SHA256"

BUILDKIT_IMAGE="$BUILDKIT_PREFIX:$BUILDKIT_SUFFIX"


# Ensure that we're using the correct builder, creating it if it doesn't exist
BUILDER_NAME="zstd-builder-$(echo $BUILDKIT_SUFFIX | sed -e 's/@/_/' -e 's/:/-/')"
if ! sudo docker buildx ls | grep -q "$BUILDER_NAME"; then
    echo "Did not find builder $BUILDER_NAME, creating it"
    sudo docker buildx create \
        --name "$BUILDER_NAME" \
        --driver docker-container \
        --driver-opt image="$BUILDKIT_IMAGE"
fi

DIR=$(dirname "${BASH_SOURCE[0]}")

# Order matters!!!
INCLUDE_DIR="$DIR/bake"
INCLUDES=(
    "utils.hcl"
    "apt.hcl"
    "pip.hcl"
    "llvm.hcl"
    "linker.hcl"
    "cmake.hcl"
)
# Prepend each element of INCLUDES with "-f $INCLUDE_DIR"
INCLUDE_FLAGS=()
for i in "${INCLUDES[@]}"; do
    INCLUDE_FLAGS+=("-f $INCLUDE_DIR/$i")
done
INCLUDE_STRING=$(
    IFS=' '
    echo "${INCLUDE_FLAGS[*]}"
)

OUTPUT_FLAGS=(
    "type=image"
    "oci-mediatypes=true"
    "compression=zstd"
    "compression-level=22"
    "force-compression=true"
    "push=false"
)
OUTPUT_STRING="--set *.output=$(
    IFS=,
    echo "${OUTPUT_FLAGS[*]}"
)"

TARGET_DIR="$DIR/target"
TARGETS=(
    "source_getter"
    "cpp_builder"
    "cuda_builder"
    "mimalloc"
    "mold"
    "cpython"
    "common_python_packages"
    "magma"
    "torch"
    "torchvision"
    "triton"
    "bsrt"
)
# Prepend each element of TARGETS with "-f $TARGET_DIR"
TARGET_FLAGS=()
for i in "${TARGETS[@]}"; do
    TARGET_FLAGS+=("-f $TARGET_DIR/$i.hcl")
done
TARGET_STRING=$(
    IFS=' '
    echo "${TARGET_FLAGS[*]}"
)

# Disable SC2086 because we want to pass the arguments as-is
# shellcheck disable=SC2086
sudo docker buildx bake \
    --builder "$BUILDER_NAME" \
    $OUTPUT_STRING \
    $INCLUDE_STRING \
    $TARGET_STRING \
    "$@"

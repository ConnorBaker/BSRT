# Build

## Docker

To build and push the docker image, run the following command:

```bash
bash docker_build.sh
```

The following environment variables can be set to customize the build:

- `CONDA_ENV_FILE`
  - Default: `conda/environment-nightly.yml`
- `MICROMAMBA_VERSION`
  - Default: `0.27.0`
- `OS_NAME`
  - Default: `jammy`
- `WITH_CUDA`
  - Default: `true`
- `CUDA_VERSION`
  - Default: `11.7.1`

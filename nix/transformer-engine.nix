{
  cmake,
  fetchFromGitHub,
  lib,
  ninja,
  symlinkJoin,
  which,
  buildPythonPackage,
  torch,
  pip,
  wheel,
  setuptools,
  python,
  pybind11,
  pydantic,
  flash-attention,
}:
assert lib.asserts.assertMsg
(torch.cudaSupport)
"tiny-cuda-nn requires torch to be built with cudaSupport"; let
  inherit (lib) strings;
  inherit (torch) cudaPackages cudaSupport;
  inherit (cudaPackages) cudatoolkit cudaFlags;

  cuda-redist = symlinkJoin {
    name = "cuda-redist";
    paths = with cudaPackages; [
      cuda_cudart
      cuda_nvcc
      cuda_nvtx
      cuda_profiler_api
      cuda_cccl
      libcublas
      libcusparse
      libcusolver
      cudnn
    ];
  };
in
  buildPythonPackage {
    name = "transformer-engine";
    version = "0.6";

    format = "setuptools";

    src = fetchFromGitHub {
      owner = "NVIDIA";
      repo = "TransformerEngine";
      # rev = "v0.5";
      # hash = "sha256-70JWg6T/3itQK86jRSE6zQj1cYlyI58AhjsiwSipu08=";
      rev = "7d6c1d02dda2d9e64e1b3548e1e99ad7757824e4";
      hash = "sha256-OHR81kuQG74n1iVWhITrc4roJ8FaeZkvYLzuMjQtXgo=";
      fetchSubmodules = true;
    };

    nativeBuildInputs = [
      which
      cmake
      ninja
      pip
      wheel
      setuptools
    ];

    buildInputs = [
      cuda-redist
      python
      pybind11
    ];

    # TODO: OpenCV, whicih also provides python bindings and supports CUDA, adds the cuda
    #   redistributable to propagatedBuildInputs. Should we do this here as well?
    propagatedBuildInputs = [
      torch
      pydantic
      flash-attention
    ];

    # NOTE: We cannot use pythonImportsCheck for this module because it uses torch to immediately
    #   initailize CUDA. We cannot assume that at the time we run the check phase, the user has an
    #   NVIDIA GPU available.
    # There are no tests for the C++ library or the python bindings, so we just skip the check
    # phase.
    doCheck = false;

    preConfigure = ''
      export CUDA_HOME=${cuda-redist}
      export CC=${cudatoolkit.cc}/bin/cc
      export CXX=${cudatoolkit.cc}/bin/c++
      export CUDAHOSTCXX=${cudatoolkit.cc}/bin/c++
      export TORCH_CUDA_ARCH_LIST=${strings.concatStringsSep ";" cudaFlags.cudaCapabilities}
      sed -i setup.py -e 's| @ git+https://github.com/ksivaman/flash-attention.git@hopper||g'
    '';

    dontUseCmakeConfigure = true;

    passthru = {
      inherit cudaPackages;
    };

    meta = with lib; {
      description = "A library for accelerating Transformer models on NVIDIA GPUs, including using 8-bit floating point (FP8) precision on Hopper GPUs, to provide better performance with lower memory utilization in both training and inference.";
      homepage = "https://github.com/NVIDIA/TransformerEngine";
      license = licenses.bsd3;
      maintainers = with maintainers; [];
      platforms = platforms.linux;
      broken = !cudaSupport;
    };
  }

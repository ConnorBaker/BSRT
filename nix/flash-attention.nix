{
  cmake,
  fetchFromGitHub,
  lib,
  ninja,
  git,
  symlinkJoin,
  which,
  buildPythonPackage,
  torch,
  pip,
  wheel,
  setuptools,
  python,
  pybind11,
}:
assert lib.asserts.assertMsg
(torch.cudaSupport)
"flash-attention requires torch to be built with cudaSupport"; let
  inherit (lib) strings;
  inherit (torch) cudaPackages cudaSupport;
  inherit (cudaPackages) cudatoolkit cudaFlags;

  cuda-redist = symlinkJoin {
    name = "cuda-redist";
    paths = with cudaPackages; [
      cuda_cudart
      cuda_nvcc
      cuda_cccl
      libcublas
      libcusparse
      libcusolver
    ];
  };
in
  buildPythonPackage {
    name = "flash-attention";
    version = "0.2.9";

    format = "setuptools";

    src = fetchFromGitHub {
      owner = "ksivaman";
      repo = "flash-attention";
      rev = "7f1c81a5f33e18f22d1e2ba42681f0743bf58b92";
      fetchSubmodules = true;
      hash = "sha256-4cmnagGNM2gW0Dq2B+q9ZwcnxGx6zNkatjJpIDWcz1E=";
    };

    nativeBuildInputs = [
      which
      cmake
      ninja
      git
      pip
      wheel
      setuptools
    ];

    buildInputs = [
      cuda-redist
      python
      pybind11
    ];

    propagatedBuildInputs = [
      torch
    ];

    doCheck = false;
    pythonImportsCheck = ["flash_attn"];
    dontUseCmakeConfigure = true;

    preConfigure = ''
      export CUDA_HOME=${cuda-redist}
      export CC=${cudatoolkit.cc}/bin/cc
      export CXX=${cudatoolkit.cc}/bin/c++
      export CUDAHOSTCXX=${cudatoolkit.cc}/bin/c++
      export TORCH_CUDA_ARCH_LIST=${strings.concatStringsSep ";" cudaFlags.cudaCapabilities}
    '';

    passthru = {
      inherit cudaPackages;
    };

    meta = with lib; {
      description = "Fast and memory-efficient exact attention";
      homepage = "https://github.com/ksivaman/flash-attention";
      license = licenses.bsd3;
      maintainers = with maintainers; [];
      platforms = platforms.linux;
      broken = !cudaSupport;
    };
  }

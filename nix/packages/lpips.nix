{
  buildPythonPackage,
  fetchFromGitHub,
  fetchpatch,
  # nativeBuildInputs
  packaging,
  # propagatedBuildInputs
  torch,
  torchvision,
  numpy,
  scipy,
  tqdm,
}: let
  attrs = {
    pname = "lpips";
    version = "0.1.4";
    format = "setuptools";

    src = fetchFromGitHub {
      owner = "richzhang";
      repo = "PerceptualSimilarity";
      rev = "v${attrs.version}";
      hash = "sha256-dIQ9B/HV/2kUnXLXNxAZKHmv/Xv37kl2n6+8IfwIALE=";
    };

    patches = [
      # update for torchvision 0.13 (backward compatible)
      (fetchpatch {
        url = "https://github.com/richzhang/PerceptualSimilarity/pull/114/commits/9606c353e5777fc95186a5a4c876e7c6a17651a6.patch";
        hash = "sha256-CVQWbYW68k4mdk+uu7HbtTQ3EmnfU7iKHXrxmBG2OzA=";
      })
      # add batch support
      (fetchpatch {
        url = "https://github.com/richzhang/PerceptualSimilarity/pull/124/commits/9539d8b49774a05f4fe3f6b8e724e45f8435038d.patch";
        hash = "sha256-YbIZVNEXKZD91NGO1b8kynBIv0aBhcluG6+CaTV5clk=";
      })
      # Fixed an Error when training L2 and SSIM models
      (fetchpatch {
        url = "https://github.com/richzhang/PerceptualSimilarity/pull/119/commits/8be9badf8df76736fd286d75445e174eb0e59dd6.patch";
        hash = "sha256-pEz1VpTttBVS1Vg2fjq7gRy4m4H3BhcNfAuGrcVch8Q=";
      })
    ];

    doCheck = false;

    nativeBuildInputs = [
      packaging
    ];

    propagatedBuildInputs = [
      torch
      torchvision
      numpy
      scipy
      tqdm
    ];

    pythonImportsCheck = [attrs.pname];
  };
in
  buildPythonPackage attrs

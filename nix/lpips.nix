{
  buildPythonPackage,
  fetchFromGitHub,
  fetchpatch,
  # Propagated build inputs
  # main deps
  numpy,
  packaging,
  scipy,
  torch,
  torchvision,
  tqdm,
}:
let
pname = "lpips";
version = "0.1.4";
in

buildPythonPackage {
  inherit pname version;

  src = fetchFromGitHub {
    owner = "richzhang";
    repo = "PerceptualSimilarity";
    rev = "v${version}";
    hash = "sha256-dIQ9B/HV/2kUnXLXNxAZKHmv/Xv37kl2n6+8IfwIALE=";
  };

  # Hopefully if a new release is cut we could get rid of all these.
  patches = [
    # Patches from upstream
    (fetchpatch {
      name = "remove-duplicate-functions.patch";
      url = "https://github.com/richzhang/PerceptualSimilarity/commit/6c5877c669f5746e6258c47fd96d0629b8a9a860.patch";
      hash = "sha256-D5B51a1mvbTB2AnRtkoUpMy2nQUiX/DRKtbWSp2icEE=";
    })
    (fetchpatch {
      name = "remove-unused-lab-functions.patch";
      url = "https://github.com/richzhang/PerceptualSimilarity/commit/ecfae6f09f9d3dddc17c88702174b7f78f9925c0.patch";
      hash = "sha256-9qqEblfCX4GRlBm8M/h8uGsElWOF0cfg5XXsMCJFWsY=";
    })
    (fetchpatch {
      name = "Update-lpips.py.patch";
      url = "https://github.com/richzhang/PerceptualSimilarity/commit/c82f1ed07d9b5004b4b41ae7b07ec53468ba3e40.patch";
      hash = "sha256-x0GH2oPflK+Ea/s4tNT/jELSJVvw56vz/yttycNcw0c=";
    })
    (fetchpatch {
      name = "Update-__init__.py.patch";
      url = "https://github.com/richzhang/PerceptualSimilarity/commit/31bc1271ae6f13b7e281b9959ac24a5e8f2ed522.patch";
      hash = "sha256-Ya56VAM+fKZ/2WPUi//opFp//5wkJE5+eAbXFCZUTpo=";
    })
    # Patches from PRs
    (fetchpatch {
      name = "Update-for-TorchVision-0.13.patch";
      url = "https://github.com/richzhang/PerceptualSimilarity/pull/114.patch";
      hash = "sha256-CVQWbYW68k4mdk+uu7HbtTQ3EmnfU7iKHXrxmBG2OzA=";
    })
    (fetchpatch {
      name = "Fixed-where-the-square-to-preserve-distance-properties.patch";
      url = "https://github.com/richzhang/PerceptualSimilarity/pull/73.patch";
      hash = "sha256-8FK6GOQp+vRsU+Hbomxpue5LxqaMjoDgN/XBjlt9KOg=";
    })
  ];

  doCheck = false;

  propagatedBuildInputs = [
    numpy
    packaging
    scipy
    torch
    torchvision
    tqdm
  ];

  pythonImportsCheck = [pname];
}

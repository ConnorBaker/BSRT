{
  python3,
  fetchFromGitHub,
  buildPythonPackage,
  # Propagated build inputs
  # main deps
  torch,
  torchvision,
  opencv4,
  mfsr_utils,
  pytorch-lightning,
  lpips,
  torchmetrics,
  wandb,
  # tune
  syne-tune,
  # lint
  black,
  flake8,
  isort,
  # typecheck
  pyright,
  mypy,
  # test
  pytest,
  pytest-xdist,
  hypothesis,
  hypothesis_torch_utils,
}:
buildPythonPackage {
  name = "bsrt";
  src = ./..;
  format = "pyproject";

  propagatedBuildInputs = [
    torch
    torchvision
    opencv4
    mfsr_utils
    pytorch-lightning
    lpips
    torchmetrics
    wandb
  ];

  passthru.optional-dependencies = {
    tune = [
      syne-tune
      syne-tune.optional-dependencies.gpsearchers
    ];
    lint = [
      black
      flake8
      isort
    ];
    typecheck = [pyright mypy];
    test = [
      pytest
      pytest-xdist
      hypothesis
      hypothesis_torch_utils
    ];
  };

  pythonImportsCheck = ["bsrt"];
}

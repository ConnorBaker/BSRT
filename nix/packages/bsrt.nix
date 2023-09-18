{
  buildPythonPackage,
  lib,
  stdenv,
  # propagatedBuildInputs
  lpips,
  mfsr_utils,
  openai-triton,
  opencv4,
  pydantic,
  pytorch-lightning,
  torch,
  torchmetrics,
  torchvision,
  wandb,
  # optional-dependencies
  black,
  mypy,
  pyright,
  ruff,
  syne-tune,
}: let
  attrs = {
    pname = "bsrt";
    version = "0.1.0";
    format = "flit";

    src = lib.sources.sourceByRegex ../.. [
      "${attrs.pname}(:?/.*)?"
      "pyproject.toml"
    ];

    doCheck = false;

    propagatedBuildInputs = [
      lpips
      mfsr_utils
      openai-triton
      opencv4
      pydantic
      pytorch-lightning
      stdenv.cc # When building with openai-triton, we need a CPP compiler
      syne-tune
      syne-tune.optional-dependencies.gpsearchers
      torch
      torchmetrics
      torchvision
      wandb
    ];

    passthru.optional-dependencies.dev = [
      # Linters/formatters
      black
      ruff
      # Type checkers
      pyright
      mypy
    ];

    pythonImportsCheck = [attrs.pname];
  };
in
  buildPythonPackage attrs

{
  fetchFromGitHub,
  buildPythonPackage,
  lib,
  pythonRelaxDepsHook,
  # Propagated build inputs
  # main deps
  packaging,
  dill,
  numpy,
  pandas,
  sortedcontainers,
  typing-extensions,
  # raytune deps
  ray,
  scikit-learn,
  scikit-optimize,
  # bore deps
  xgboost,
  gpy,
  # kde deps
  statsmodels,
  # gpsearchers deps
  scipy,
  autograd,
  # aws deps
  boto3,
  sagemaker,
  pyyaml,
  ujson,
  s3fs,
}: let
  pname = "syne-tune";
  version = "0.6.0";
in
  buildPythonPackage {
    inherit pname version;

    src = fetchFromGitHub {
      owner = "awslabs";
      repo = pname;
      rev = "v${version}";
      hash = "sha256-eP80CBRUbspuUzB37qm7sPKcVITrgn59/pNjYo6fkEc=";
    };

    doCheck = false;

    nativeBuildInputs = [pythonRelaxDepsHook];

    pythonRelaxDeps = ["numpy"];

    propagatedBuildInputs = [
      packaging
      dill
      numpy
      pandas
      typing-extensions
      sortedcontainers
    ];

    pythonImportsCheck = [(lib.strings.replaceStrings ["-"] ["_"] pname)];

    passthru.optional-dependencies = {
      raytune = [
        ray
        ray.optional-dependencies.tune
        scikit-learn
        scikit-optimize
      ];
      bore = [
        numpy
        xgboost
        scikit-learn
        gpy
      ];
      kde = [statsmodels];
      gpsearchers = [
        scipy
        autograd
      ];
      # benchmarks = required_benchmarks;
      # blackbox-repository = required_blackbox_repository;
      aws = [
        boto3
        sagemaker
        pyyaml
        ujson
        s3fs
      ];
      # yahpo = required_yahpo;
      # dev = required_dev;
      # extra = required_extra;
    };
  }

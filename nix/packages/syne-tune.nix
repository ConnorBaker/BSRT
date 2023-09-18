{
  fetchFromGitHub,
  buildPythonPackage,
  # nativeBuildInputs
  packaging,
  # propagatedBuildInputs
  dill,
  numpy,
  pandas,
  typing-extensions,
  # passthru.optional-dependencies
  autograd,
  boto3,
  botorch,
  gpy,
  matplotlib,
  pyyaml,
  ray,
  s3fs,
  sagemaker,
  scikit-learn,
  scikit-optimize,
  scipy,
  statsmodels,
  ujson,
  xgboost,
}: let
  attrs = {
    pname = "syne_tune";
    version = "0.9.1";
    format = "setuptools";

    src = fetchFromGitHub {
      owner = "awslabs";
      repo = "syne-tune";
      rev = "v${attrs.version}";
      sha256 = "sha256-Dyu6ZkMkTNkSRHx2QTBGXwn2lyy3YuqRqBm0Hj8tXZM=";
    };

    doCheck = false;

    nativeBuildInputs = [
      packaging
    ];

    propagatedBuildInputs = [
      dill
      numpy
      pandas
      typing-extensions
    ];

    pythonImportsCheck = [attrs.pname];

    passthru.optional-dependencies = {
      aws = [
        boto3
        pyyaml
        s3fs
        sagemaker
        ujson
      ];
      bore = [
        gpy
        numpy
        scikit-learn
        xgboost
      ];
      botorch = [
        botorch
      ];
      # dev = [];
      gpsearchers = [
        autograd
        scipy
      ];
      kde = [statsmodels];
      raytune = [
        ray
        ray.optional-dependencies.tune
        scikit-learn
        scikit-optimize
      ];
      sklearn = [scikit-learn];
      visual = [matplotlib];
    };
  };
in
  buildPythonPackage attrs

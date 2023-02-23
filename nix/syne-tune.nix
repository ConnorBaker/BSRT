{
  python,
  fetchFromGitHub,
  buildPythonPackage,
  # Propagated build inputs
  # main deps
  packaging,
  dill,
  numpy,
  pandas,
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
}:
buildPythonPackage {
  name = "syne_tune";
  version = "0.1.4";

  src = fetchFromGitHub {
    owner = "awslabs";
    repo = "syne-tune";
    rev = "e6036878aa7338f39467b5540537605f72df75c9";
    hash = "sha256-7mk6xlIKgllNVLOop+GBTS24b9+HKyCq5EZVHlBxZ8c=";
  };

  doCheck = false;

  propagatedBuildInputs = [
    packaging
    dill
    numpy
    pandas
    typing-extensions
  ];

  pythonImportsCheck = ["syne_tune"];

  # extras_require={
  #       "raytune": required_ray,
  #       "bore": required_bore,
  #       "kde": required_kde,
  #       "gpsearchers": required_gpsearchers,
  #       "benchmarks": required_benchmarks,
  #       "blackbox-repository": required_blackbox_repository,
  #       "aws": required_aws,
  #       "yahpo": required_yahpo,
  #       "dev": required_dev,
  #       "extra": required_extra,
  #   },

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

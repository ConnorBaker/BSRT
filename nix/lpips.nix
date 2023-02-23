{
  python,
  fetchFromGitHub,
  buildPythonPackage,
  # Propagated build inputs
  # main deps
  packaging,
  torch,
  torchvision,
  numpy,
  scipy,
  tqdm,
}:
buildPythonPackage {
  name = "lpips";
  version = "0.1.4";

  src = fetchFromGitHub {
    owner = "ercanburak";
    repo = "PerceptualSimilarity";
    rev = "9606c353e5777fc95186a5a4c876e7c6a17651a6";
    hash = "sha256-Vte5wfGIcrC08ub0PdbdgPzxFARqCkPnbfbqAdXRp9c=";
  };

  doCheck = false;

  propagatedBuildInputs = [
    packaging
    torch
    torchvision
    numpy
    scipy
    tqdm
  ];

  pythonImportsCheck = ["lpips"];
}

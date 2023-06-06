final: prev: {
  python3Packages = prev.python3Packages.overrideScope (python-final: python-prev: {
    syne-tune = python-final.callPackage ../syne-tune.nix {};
    lpips = python-final.callPackage ../lpips.nix {};
    bsrt = python-final.callPackage ../bsrt.nix {};
    # transformer-engine = python-final.callPackage ../transformer-engine.nix {};
    # flash-attention = python-final.callPackage ../flash-attention.nix {};
  });
}

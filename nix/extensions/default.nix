final: prev: let
  inherit (prev.lib) attrsets;
  new.python310.pkgs = prev.python310.pkgs.overrideScope (python-final: python-prev: {
    syne-tune = python-final.callPackage ../syne-tune.nix {};
    lpips = python-final.callPackage ../lpips.nix {};
    bsrt = python-final.callPackage ../bsrt.nix {};
  });
in
  attrsets.recursiveUpdate prev new

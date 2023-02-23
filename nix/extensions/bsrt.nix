final: prev: let
  inherit (prev.lib) attrsets;
  python-overlay = python-final: python-prev: {
    lpips = python-final.callPackage ../lpips.nix {};
    bsrt = python-final.callPackage ../default.nix {};
  };
  new.python310.pkgs = prev.python310.pkgs.overrideScope python-overlay;
in
  attrsets.recursiveUpdate prev new

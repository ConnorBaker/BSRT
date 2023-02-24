final: prev: let
  inherit (prev.lib) attrsets;
  new.opencv4 = prev.callPackage ../opencv4 {
    cudnnSupport = true;
    enablePython = true;
    pythonPackages = prev.python310.pkgs;
    inherit (prev.darwin.apple_sdk.frameworks) AVFoundation Cocoa VideoDecodeAcceleration CoreMedia MediaToolbox;
    ffmpeg = prev.ffmpeg_4;
  };
  new.opencv = new.opencv4;
  new.python310.pkgs = prev.python310.pkgs.overrideScope (python-final: python-prev: {
    inherit (new) opencv4;
    syne-tune = python-final.callPackage ../syne-tune.nix {};
    lpips = python-final.callPackage ../lpips.nix {};
    bsrt = python-final.callPackage ../bsrt.nix {};
  });
in
  attrsets.recursiveUpdate prev new

{
  # Note: How do we set them to follow nixpkgs transitively?
  #   Example: mfsr_utils depends on hypothesis_torch_utils, and we can set mfsr_utils to follow
  #   nixpkgs, but how do we set hypothesis_torch_utils to follow nixpkgs?
  #
  #   I've just listed it here, but with many dependencies, this will get out of hand. Also
  #   requires knowing all transitive dependencies ahead of time.
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs";

    opencv-fix.url = "github:connorbaker/nixpkgs/feat/opencv-use-cudaPackages";

    hypothesis_torch_utils.url = "github:connorbaker/hypothesis_torch_utils";
    hypothesis_torch_utils.inputs.nixpkgs.follows = "nixpkgs";
    mfsr_utils.url = "github:connorbaker/mfsr_utils";
    mfsr_utils.inputs.nixpkgs.follows = "nixpkgs";
    mfsr_utils.inputs.hypothesis_torch_utils.follows = "hypothesis_torch_utils";
    nixGL.url = "github:guibou/nixGL";
    nixGL.inputs.nixpkgs.follows = "nixpkgs";
  };

  nixConfig = {
    # Add the CUDA maintainer's cache to the binary cache list.
    extra-substituters = [
      "https://cuda-maintainers.cachix.org"
    ];
    extra-trusted-public-keys = [
      "cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
    ];
  };

  outputs = {
    self,
    nixpkgs,
    opencv-fix,
    hypothesis_torch_utils,
    mfsr_utils,
    nixGL,
  }: let
    system = "x86_64-linux";
    nvidiaDriver = {
      version = "525.89.02";
      sha256 = "sha256-DkEsiMW9mPhCqDmm9kYU8g5MCVDvfP+xKxWKcWM1k+k=";
    };

    overlay = nixpkgs.lib.composeManyExtensions [
      (final: prev: {
        python3 = prev.python310;
        python3Packages = prev.python310Packages;
        cudaPackages = prev.cudaPackages_11_8;
      })
      # Bugfix: in the `cudaPackages` 11.7 to 11.8 transition, be aware that `cuda_profiler_api.h` 
      #   is no longer in `cuda_nvprof`; it's in a new `cuda_profiler_api` package in 
      #   `cudaPackages`.
      (final: prev: {
        magma = prev.magma.overrideAttrs (oldAttrs: {
          buildInputs = oldAttrs.buildInputs ++ [ prev.cudaPackages.cuda_profiler_api ];
        });
      })
      # Bugfix: OpenCV doesn't use the an NVCC-compatible compiler by default, so we need to
      #   override it.
      (final: prev: {
        opencv4 = prev.callPackage "${opencv-fix}/pkgs/development/libraries/opencv/4.x.nix" {
          inherit (prev.darwin.apple_sdk.frameworks) AVFoundation Cocoa VideoDecodeAcceleration CoreMedia MediaToolbox;
          enablePython = true;
          pythonPackages = prev.python3Packages;
          ffmpeg = prev.ffmpeg_4;
        };
      })
      nixGL.overlays.default
      mfsr_utils.overlays.default
      (import ./nix/extensions)
    ];

    pkgs = import nixpkgs {
      inherit system;
      config = {
        allowUnfree = true;
        cudaSupport = true;
        cudaCapabilities = ["8.0"];
        cudaForwardCompat = false;
      };
      overlays = [overlay];
    };

    inherit (pkgs) python310 python310Packages nixgl;
    inherit (python310Packages) bsrt;
    inherit (nixgl.nvidiaPackages nvidiaDriver) nixGLNvidia;
  in {
    inherit pkgs;
    overlays.default = overlay;
    packages.${system}.default = pkgs.mkShell {
      packages =
        [
          python310
          bsrt.propagatedBuildInputs
          nixGLNvidia
        ]
        ++ (
          with bsrt.passthru.optional-dependencies;
            tune ++ lint ++ typecheck ++ test
        );

      # Make an alias for python so it's wrapped with nixGLNvidia.
      shellHook = ''
        alias python3="${nixGLNvidia.name} python3"
        alias python="${nixGLNvidia.name} python3"
      '';
    };

    formatter.${system} = pkgs.alejandra;
  };
}

{
  inputs.nixpkgs.url = "github:nixos/nixpkgs";
  inputs.mfsr_utils.url = "github:connorbaker/mfsr_utils";
  inputs.nixGL.url = "github:guibou/nixGL";
  inputs.nixGL.inputs.nixpkgs.follows = "nixpkgs";

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

    inherit (pkgs) python310 python310Packages nixgl cudaPackages;
    inherit (python310Packages) bsrt;
    inherit (nixgl.nvidiaPackages nvidiaDriver) nixGLNvidia;
  in {
    inherit pkgs;
    overlays.default = overlay;
    packages.${system}.default = pkgs.mkShell {
      packages =
        [
          python310
          bsrt
          nixGLNvidia
        ]
        ++ (
          with bsrt.passthru.optional-dependencies;
            tune ++ lint ++ typecheck ++ test
        );

      # Make an alias for python so it's wrapped with nixGLNvidia-525.89.02.
      shellHook = ''
        alias python3="nixGLNvidia-525.89.02 python3"
        alias python="nixGLNvidia-525.89.02 python3"
      '';
    };

    formatter.${system} = pkgs.alejandra;
  };
}

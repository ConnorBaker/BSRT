{
  inputs = {
    # nixpkgs.url = "github:connorbaker/nixpkgs/feat/cudnn_8_9_1";
    nixpkgs.url = "github:ConnorBaker/nixpkgs/feat/xformers_0_0_20";
    nixGL = {
      url = "github:guibou/nixGL";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    mfsr_utils.url = "github:connorbaker/mfsr_utils";
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
    nixGL,
    mfsr_utils,
  }: let
    system = "x86_64-linux";

    overlay = nixpkgs.lib.composeManyExtensions [
      mfsr_utils.overlays.default
      (import ./nix/extensions)
    ];

    pkgs = import nixpkgs {
      inherit system;
      config = {
        allowUnfree = true;
        cudaSupport = true;
        cudaCapabilities = ["8.9"];
        cudaForwardCompat = false;
      };
      # TODO: Getting CUDA errors when using cudaPackages_11_8 -- try to reproduce with 11_7.
      overlays = [
        (final: prev: {
          python3 = prev.python310;
          python3Packages = prev.python310Packages;
          cudaPackages = prev.cudaPackages_11_8;
        })
        (final: prev: {
          nixgl = import nixGL.outPath {
            pkgs = prev;
            enableIntelX86Extensions = true;
            enable32bits = false;
            nvidiaVersion = "530.41.03";
            nvidiaHash = "sha256-riehapaMhVA/XRYd2jQ8FgJhKwJfSu4V+S4uoKy3hLE=";
          };
        })
        overlay
      ];
    };
  in {
    overlays.default = overlay;
    legacyPackages.${system} = pkgs;
    devShells.${system}.default = pkgs.mkShell {
      inputsFrom = [pkgs.python3Packages.bsrt];
      packages =
        [
          pkgs.nixgl.nixGLNvidia
          pkgs.python3
          pkgs.python3Packages.xformers
          # TODO: Must set torch._inductor.config.compile_threads = 1 or else Segmentation fault
          #   when running torch.compile.
          pkgs.python3Packages.bsrt.propagatedBuildInputs
        ]
        ++ (
          with pkgs.python3Packages.bsrt.passthru.optional-dependencies;
            tune ++ lint ++ typecheck ++ test
        );

      # Make an alias for python so it's wrapped with nixGLNvidia.
      shellHook = ''
        alias python3="${pkgs.nixgl.nixGLNvidia.name} python3"
        alias python="${pkgs.nixgl.nixGLNvidia.name} python3"
      '';
    };

    formatter.${system} = pkgs.alejandra;
  };
}

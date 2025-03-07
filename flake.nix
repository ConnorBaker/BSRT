{
  inputs = {
    flake-parts = {
      inputs.nixpkgs-lib.follows = "nixpkgs";
      url = "github:hercules-ci/flake-parts";
    };
    flake-utils.url = "github:numtide/flake-utils";
    mfsr_utils = {
      url = "github:ConnorBaker/mfsr_utils";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    nixpkgs.url = "github:NixOS/nixpkgs";
    nixpkgs-lib.url = "github:nix-community/nixpkgs.lib";
    pre-commit-hooks-nix = {
      inputs = {
        flake-utils.follows = "flake-utils";
        nixpkgs-stable.follows = "nixpkgs";
        nixpkgs.follows = "nixpkgs";
      };
      url = "github:cachix/pre-commit-hooks.nix";
    };
    treefmt-nix = {
      inputs.nixpkgs.follows = "nixpkgs";
      url = "github:numtide/treefmt-nix";
    };
  };

  nixConfig = {
    extra-substituters = [
      "https://cuda-maintainers.cachix.org"
    ];
    extra-trusted-substituters = [
      "https://cuda-maintainers.cachix.org"
    ];
    extra-trusted-public-keys = [
      "cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
    ];
  };

  outputs = inputs:
    inputs.flake-parts.lib.mkFlake {inherit inputs;} {
      systems = ["x86_64-linux"];
      imports = [
        inputs.treefmt-nix.flakeModule
        inputs.pre-commit-hooks-nix.flakeModule
        ./nix
      ];

      perSystem = {
        config,
        pkgs,
        ...
      }: {
        pre-commit.settings = {
          hooks = {
            # Formatter checks
            treefmt.enable = true;

            # Nix checks
            deadnix.enable = true;
            nil.enable = true;
            statix.enable = true;

            # Python checks
            mypy.enable = true;
            pyright.enable = true;
            ruff.enable = true; # Ruff both lints and checks sorted imports
          };
          settings = let
            # We need to provide wrapped version of mypy and pyright which can find our imports.
            # TODO: The script we're sourcing is an implementation detail of `mkShell` and we should
            # not depend on it exisitng. In fact, the first few lines of the file state as much
            # (that's why we need to strip them, sourcing only the content of the script).
            wrapper = name:
              pkgs.writeShellScript name ''
                source <(sed -n '/^declare/,$p' ${config.devShells.bsrt})
                ${name} "$@"
              '';
          in {
            # Formatter
            treefmt.package = config.treefmt.build.wrapper;

            # Python
            mypy.binPath = "${wrapper "mypy"}";
            pyright.binPath = "${wrapper "pyright"}";
          };
        };

        treefmt = {
          projectRootFile = "flake.nix";
          programs = {
            # Markdown
            mdformat.enable = true;

            # Nix
            alejandra.enable = true;

            # Python
            black.enable = true;
            ruff.enable = true; # Ruff both lints and checks sorted imports

            # Shell
            shellcheck.enable = true;
            shfmt.enable = true;
          };
        };
      };
    };
}

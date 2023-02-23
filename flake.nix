{
  inputs.nixpkgs.url = "github:nixos/nixpkgs/master";
  inputs.mfsr_utils.url = "github:connorbaker/mfsr_utils";

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
  }: let
    system = "x86_64-linux";

    inherit (nixpkgs.lib) composeManyExtensions;
    overlay = composeManyExtensions [
      mfsr_utils.overlays.default
      (import ./nix/extensions/bsrt.nix)
    ];

    pkgs = import nixpkgs {
      inherit system;
      config = {
        allowUnfree = true;
        cudaSupport = true;
        cudaCapabilities = ["8.0"];
        cudaForwardCompat = true;
      };
      overlays = [overlay];
    };

    inherit (pkgs.cudaPackages) cudatoolkit;
    inherit (pkgs) python310;
    inherit (python310.pkgs) bsrt;
  in {
    overlays.default = overlay;
    devShells.${system}.default = pkgs.mkShell {
      packages = [
        cudatoolkit
        python310
        bsrt
      ];
      shellHook = ''
        export CUDA_HOME=${cudatoolkit}
        export PATH=$CUDA_HOME/bin:$PATH
        if [ ! -f /run/opengl-driver/lib/libcuda.so.1 ]; then
          echo
          echo "Could not find /run/opengl-driver/lib/libcuda.so.1."
          echo
          echo "You have at least three options:"
          echo
          echo "1. Use nixGL (https://github.com/guibou/nixGL)"
          echo "2. Add libcuda.so.1 to your LD_PRELOAD environment variable."
          echo "3. Symlink libcuda.so.1 to /run/opengl-driver/lib/libcuda.so.1."
          echo
          echo "   This is the easiest option, but it requires root."
          echo "   You can do this by running:"
          echo
          echo "   sudo mkdir -p /run/opengl-driver/lib"
          echo "   sudo ln -s /usr/lib64/libcuda.so.1 /run/opengl-driver/lib/libcuda.so.1"
          echo
          echo "Continuing to the shell, but be aware that CUDA might not work."
          echo
        fi
      '';
    };

    formatter.${system} = pkgs.alejandra;
  };
}

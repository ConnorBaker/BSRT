{
  perSystem = {
    config,
    inputs',
    pkgs,
    ...
  }: {
    packages = {
      bsrt = pkgs.python3Packages.callPackage ./bsrt.nix {
        inherit (inputs'.mfsr_utils.packages) mfsr_utils;
        inherit (config.packages) lpips pydantic syne-tune;
      };
      lpips = pkgs.python3Packages.callPackage ./lpips.nix {};
      pydantic = pkgs.python3Packages.callPackage ./pydantic.nix {};
      syne-tune = pkgs.python3Packages.callPackage ./syne-tune.nix {};
    };
  };
}

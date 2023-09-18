{
  perSystem = {
    config,
    pkgs,
    ...
  }: {
    devShells = {
      bsrt = pkgs.mkShell {
        inputsFrom = [config.packages.bsrt];
        packages = with config.packages.bsrt.optional-dependencies; [dev];
      };
      default = pkgs.mkShell {
        packages = [config.packages.bsrt pkgs.python3Packages.jupyterlab];
      };
    };
  };
}

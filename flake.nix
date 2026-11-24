{
  description = "Matterstate simulator kernels and external reference tests";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-parts = {
      url = "github:hercules-ci/flake-parts";
      inputs.nixpkgs-lib.follows = "nixpkgs";
    };
    devshell = {
      url = "github:numtide/devshell";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [ inputs.devshell.flakeModule ];
      systems = [ "x86_64-linux" ];
      perSystem = { pkgs, ... }: {
        formatter = pkgs.nixfmt;
        devshells.default = {
          packages = [
            pkgs.actionlint
            pkgs.ast-grep
            pkgs.nixfmt
            pkgs.python313
            pkgs.ruff
            pkgs.uv
          ];
          env = [
            {
              name = "OGBENCH_PYTHON";
              value = pkgs.python311.interpreter;
            }
            {
              name = "UV_PYTHON";
              value = pkgs.python313.interpreter;
            }
            {
              name = "UV_PYTHON_DOWNLOADS";
              value = "never";
            }
            {
              name = "PYTHONPATH";
              unset = true;
            }
            {
              name = "LD_LIBRARY_PATH";
              value = pkgs.lib.makeLibraryPath [
                pkgs.stdenv.cc.cc.lib
                pkgs.zlib
              ];
            }
          ];
          commands = [
            {
              name = "check";
              help = "Verify source, extraction snapshots, and external parity";
              command = ''
                set -eu
                actionlint
                nixfmt --check flake.nix
                ruff check .
                ruff format --check .
                ast-grep test
                ast-grep scan
                uv run --frozen --group reference python -m pytest -q -p no:cacheprovider tests
                uv run --project tests/reference/ogbench --frozen --python "$OGBENCH_PYTHON" python -c "import ogbench"
              '';
            }
          ];
        };
      };
    };
}

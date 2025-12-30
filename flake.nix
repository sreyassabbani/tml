{
  inputs = {
    nixpkgs.url = "https://flakehub.com/f/NixOS/nixpkgs/0.tar.gz";
    flake-utils.url = "github:numtide/flake-utils";

    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      fenix,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
        llvm = pkgs.llvmPackages_21;

        # Toolchain "set" pinned by name (date). We'll fill sha256 once Nix tells us.
        pinned = fenix.packages.${system}.fromToolchainName {
          name = "nightly-2025-12-29";
          sha256 = "sha256-ZyGYzojlRUzuDMZ2OznKWXa/eXhuQkTZ2iHlXOputws=";
        };

        # One derivation containing exactly the components we want.
        rust = pinned.withComponents [
          "cargo"
          "rustc"
          "rustfmt"
          "clippy"
          "rust-src"
        ];

        ra = fenix.packages.${system}.rust-analyzer;
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            rust
            ra
            llvm.lldb
            pkgs.gemini-cli
            pkgs.bacon
          ];

          shellHook = ''
            echo "Nix dev shell activated"
            echo "rustc: $(rustc --version 2>/dev/null || echo 'not found')"
            echo "cargo: $(cargo --version 2>/dev/null || echo 'not found')"
            echo "rust-analyzer: $(rust-analyzer --version 2>/dev/null || echo 'not found')"
          '';
        };
      }
    );
}

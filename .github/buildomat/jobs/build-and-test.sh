#!/bin/bash
#:
#: name = "build-and-test"
#: variety = "basic"
#: target = "helios"
#: rust_toolchain = "stable"
#: output_rules = [ ]
#:

set -o errexit
set -o pipefail
set -o xtrace

cargo --version
rustc --version

banner build
cargo build

cargo fmt -- --check
cargo clippy -- --deny warnings

banner test
cargo test

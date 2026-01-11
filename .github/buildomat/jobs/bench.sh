#!/bin/bash
#:
#: name = "bench"
#: variety = "basic"
#: target = "helios-2.0"
#: rust_toolchain = "stable"
#: output_rules = [ ]
#:

set -o errexit
set -o pipefail
set -o xtrace

cargo --version
rustc --version

banner bench
cargo bench

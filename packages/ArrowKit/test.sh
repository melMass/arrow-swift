#!/bin/sh
# `swift test` with MLX's shaders where MLX finds them. SwiftPM compiles
# default.metallib into mlx-swift_Cmlx.bundle, but MLX looks for that bundle
# beside the main executable (xctest's, in Xcode). Its first lookup is
# mlx.metallib beside the binary MLX is linked into, the test binary here.
# Arguments go to both swift build and swift test (--scratch-path, --filter).
# Model suites need the weights in the HF cache and ARROW_MODEL_TESTS=1.
set -e
cd "$(dirname "$0")"
build_args=""
# Suites load the 5.3 GB model; run in parallel they contend for the GPU and memory.
parallel="--no-parallel"
test_args=""
while [ $# -gt 0 ]; do
    case "$1" in
        --filter|--skip) test_args="$test_args $1 $2"; shift 2 ;;
        --parallel) parallel="--parallel"; shift ;;
        *) build_args="$build_args $1"; test_args="$test_args $1"; shift ;;
    esac
done
bin=$(swift build --show-bin-path $build_args)
library="$bin/ArrowKitTests.xctest/Contents/MacOS/mlx.metallib"
# A file added to the bundle after signing fails the next build's code signing.
rm -f "$library"
swift build --build-tests $build_args
cp "$bin/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib" "$library"
swift test --skip-build $parallel $test_args

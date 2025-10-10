# Repository Guidelines

This repository hosts a GPU‑accelerated, real‑time image stitching fork of OpenCV 2.4. Use this guide to contribute effectively and keep builds reproducible.

## Project Structure & Module Organization
- `modules/` OpenCV modules (core, imgproc, stitching, gpu, etc.).
- `src/` Top‑level integrations and CachedStitcher glue code.
- `include/` Public headers (e.g., `CachedStitcher.hpp`).
- `apps/` Binaries and CLI entry points (e.g., `realtime_stitching`).
- `samples/` Usage examples and performance demos.
- `data/` Sample assets for local runs.
- `cmake/`, `CMakeLists.txt` Build system; enable CUDA and module flags here.
- `doc/` Documentation and diagrams; update if you change behavior.

## Build, Test, and Development Commands
- Configure (Release, CUDA on):
  - `mkdir -p build && cd build`
  - `cmake .. -DCMAKE_BUILD_TYPE=Release -DWITH_CUDA=ON -DCUDA_FAST_MATH=ON -DBUILD_opencv_stitching=ON`
- Build: `cmake --build . --parallel` (or `make -j$(nproc)`).
- Run sample app: `./apps/realtime_stitching --camera 0 1` or `./apps/realtime_stitching img1.jpg img2.jpg`.
- Optional tests (if CTest is enabled): `ctest --output-on-failure`.

## Coding Style & Naming Conventions
- C++11, 4‑space indent, UTF‑8, no trailing whitespace.
- Classes: `CamelCase` (e.g., `CachedStitcher`).
- Methods/Functions: `lowerCamelCase` (e.g., `composePanoramaGPU`).
- Members: `snake_case_` with trailing underscore for private fields (e.g., `gpu_enabled_`).
- Headers include guards or `#pragma once`; keep public API in `include/`.
- Prefer RAII; avoid raw `new/delete`; use `cv::` types where applicable.
- Run formatter if configured; otherwise match surrounding style.

## Testing Guidelines
- Prefer runnable examples in `samples/` or `apps/` for feature validation.
- Provide minimal assets under `data/` (or link instructions to obtain them).
- Add performance notes (FPS, resolution, GPU model) in PR description.
- Name example/test sources descriptively: `samples/gpu/<feature>_<purpose>.cpp`.

## Commit & Pull Request Guidelines
- Commits: imperative mood, concise scope prefix when helpful (e.g., `stitching: cache x/y maps on init`).
- Keep commits focused; include build or API changes in separate commits.
- PRs must include:
  - Summary of changes and rationale.
  - Build/run instructions (`cmake` flags) and expected output.
  - Benchmarks (FPS, resolution) and environment (GPU, CUDA, driver).
  - Screenshots or short logs for visual/CLI features.
  - Linked issues and migration notes for any API changes.

## Security & Configuration Tips
- Match CUDA, driver, and `CUDA_ARCH_BIN` to target GPUs.
- Guard GPU paths with CPU fallbacks; never regress CPU builds.
- Avoid embedding large binaries; reference datasets or use `data/` locally.


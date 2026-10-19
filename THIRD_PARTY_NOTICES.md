# Third-party notices

The release includes source code needed by the evaluation runners:

- The evaluation runtime in `src/rho/` and derived runner code retain RHO
  components. The original MIT license and copyright notice are reproduced
  verbatim in `third_party/licenses/RHO-MIT.txt`. That notice attributes the
  upstream work; it does not identify the anonymous HarnessPro contributors.
- `third_party/programbench/`: ProgramBench, copyright Meta Platforms, Inc.
  and affiliates, MIT License. See its `LICENSE` file.
- `third_party/pier/`: Pier, Apache License 2.0. Pier includes code derived
  from Harbor; see its `LICENSE` and `NOTICE` files.
- `third_party/rtk-bin/rtk`: RTK 0.28.2 Linux x86-64 executable, Apache
  License 2.0. See the adjacent `LICENSE` file.

Python dependencies installed through `uv sync` are not included in the
release archive and retain their respective upstream licenses. The separate Go
toolchain distributions required by some SWE-bench Pro tasks are not bundled.

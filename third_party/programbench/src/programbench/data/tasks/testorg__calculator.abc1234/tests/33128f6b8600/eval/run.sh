#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

set -e
cd "$(dirname "$0")/.."
pip install pytest --quiet 2>/dev/null || true
python3 -m pytest eval/tests/ --junitxml=eval/results.xml -v || true

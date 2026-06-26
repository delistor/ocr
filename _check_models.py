# Copyright (c) 2025 delistor
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Check model_name from inference.yml for all existing model dirs"""
import os, yaml

targets = [
    "PP-OCRv6_medium_det_onnx",
    "PP-OCRv6_medium_rec_onnx",
    "PP-LCNet_x1_0_textline_ori_onnx",
    "UVDoc_onnx",
]

for d in targets:
    yml_path = os.path.join(d, "inference.yml")
    if os.path.isfile(yml_path):
        with open(yml_path) as f:
            cfg = yaml.safe_load(f)
        model_name = cfg.get("Global", {}).get("model_name", "?")
        print(f"{d}: model_name = {model_name}")
    else:
        print(f"{d}: NO inference.yml")
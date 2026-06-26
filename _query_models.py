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
"""Query HuggingFace API for PP-OCRv6 model list"""
import json, urllib.request, sys

# Search for all PaddlePaddle PP-OCRv6 ONNX models
url = "https://huggingface.co/api/models?search=PP-OCRv6&author=PaddlePaddle&sort=downloads&direction=-1"
try:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.load(resp)
    for m in data:
        mid = m["modelId"]
        downloads = m.get("downloads", 0)
        tags = m.get("tags", [])
        print(f"{mid:<55s} downloads={downloads:<8d} tags={','.join(tags[:5])}")
    print(f"\nTotal: {len(data)} models found")
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)

# Also search for PP-LCNet + textline/orientation
url2 = "https://huggingface.co/api/models?search=PaddlePaddle+textline+orientation&author=PaddlePaddle"
try:
    req = urllib.request.Request(url2, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp2:
        data2 = json.load(resp2)
    print("\n--- Textline/Detection orientation models ---")
    for m in data2:
        mid = m["modelId"]
        downloads = m.get("downloads", 0)
        print(f"{mid:<55s} downloads={downloads}")
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
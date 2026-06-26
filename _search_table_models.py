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
"""Search for PaddlePaddle table detection / recognition / structure models"""
import json, urllib.request

# Search all PaddlePaddle models
url = "https://huggingface.co/api/models?author=PaddlePaddle&sort=downloads&direction=-1&limit=300"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req, timeout=15) as resp:
    data = json.load(resp)

print("=" * 90)
print("ALL table-related PaddlePaddle models (sorted by downloads)")
print("=" * 90)
print(f"{'Model ID':<55s} {'Downloads':<10s}")
print("-" * 90)
for m in data:
    mid = m["modelId"]
    tags = ",".join(m.get("tags", []))
    downloads = m.get("downloads", 0)
    low_mid = mid.lower()
    low_tags = tags.lower()
    if any(kw in low_mid or kw in low_tags for kw in ["table", "slanet", "tablenet", "layoutxlm", "pp-table"]):
        print(f"{mid:<55s} {downloads:<10d}")
print()

print("=" * 90)
print("ONNX table-related models only")
print("=" * 90)
print(f"{'Model ID':<55s} {'Downloads':<10s}")
print("-" * 90)
for m in data:
    mid = m["modelId"]
    tags = ",".join(m.get("tags", []))
    downloads = m.get("downloads", 0)
    # Filter for ONNX + table/layout/doc
    has_onnx = "onnx" in mid.lower() or "onnx" in tags.lower()
    low = (mid + " " + tags).lower()
    is_table_rel = any(kw in low for kw in ["layout", "table", "doc", "slanet"])
    if has_onnx and is_table_rel:
        print(f"{mid:<55s} {downloads:<10d}")
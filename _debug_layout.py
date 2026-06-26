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
"""诊断 PP-DocLayoutV3 的 predict 输出格式"""
from paddlex import create_model
import cv2, numpy as np

m = create_model(
    model_name="PP-DocLayoutV3",
    model_dir="D:/ocr/PP-DocLayoutV3_onnx",
    engine="onnxruntime",
)
img = cv2.imread("D:/ocr/ocr_test.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

r = next(m.predict(img))
print("keys:", list(r.keys()))
print()
for k in r.keys():
    v = r[k]
    print(f"  {k}: type={type(v).__name__}")
    if isinstance(v, (list, np.ndarray)):
        print(f"       len={len(v)}")
        if len(v) > 0:
            print(f"       first type={type(v[0]).__name__}")
            print(f"       first value={v[0]}")
    else:
        print(f"       value={v}")
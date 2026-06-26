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
"""Quick test of PaddleX detection and recognition models"""
import cv2
import numpy as np
from paddlex import create_model

# Test detection
det = create_model('PP-OCRv6_medium_det', model_dir=r'D:\ocr\PP-OCRv6_medium_det_onnx', engine='onnxruntime')
for r in det.predict(r'D:\ocr\ocr_test.png'):
    polys = r.get('dt_polys')
    scores = r.get('dt_scores')
    if polys is not None:
        print(f'Detection: found {len(polys)} text boxes')
        for i, (p, s) in enumerate(zip(polys[:3], scores[:3])):
            print(f'  Box {i}: shape={p.shape}, score={s:.3f}')
    else:
        print('Detection: no boxes found')
        for k, v in r.items():
            if isinstance(v, (list, np.ndarray)):
                if hasattr(v, 'shape'):
                    print(f'  key={k}: type={type(v).__name__}, shape={v.shape}')
                else:
                    print(f'  key={k}: type={type(v).__name__}, len={len(v)}')
            else:
                print(f'  key={k}: type={type(v).__name__}, val={v}')
    break

# Test recognition
rec = create_model('PP-OCRv6_medium_rec', model_dir=r'D:\ocr\PP-OCRv6_medium_rec_onnx', engine='onnxruntime')
print('Recognition model loaded successfully')
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
"""
OCR 推理脚本（基于 PaddleX 本地 ONNX 模型）
=============================================
所有模型文件放在当前目录，自动发现加载，纯本地运行，不联网。

完备的工作流（每个模块可独立开关，按 config.json 控制）:

   📷 输入图片
      │
      ├── [可选] 文档方向分类 ── 自动旋转图片到正确方向
      │   PP-LCNet_x1_0_doc_ori_onnx
      │
      ├── [可选] 文档畸变校正 ── 展平弯曲/变形的文档图片
      │   UVDoc_onnx
      │
      ├── [可选] 版面分析 ── 检测表格、标题、图片等区域
      │   PP-DocLayoutV3_onnx
      │
      ├── 文本检测 ── 检测所有文字框位置
      │   PP-OCRv6_medium_det_onnx
      │
      ├── [可选] 文字行方向 ── 判断每行是否旋转
      │   PP-LCNet_x1_0_textline_ori_onnx
      │
      ├── 文本识别 ── 识别每个框中的文字
      │   PP-OCRv6_medium_rec_onnx
      │
      ├── [可选] 表格类型分类 ── 判断有线/无线表格
      │   PP-LCNet_x1_0_table_cls_onnx
      │
      ├── [可选] 表格结构识别 ── 解析表格行列结构
      │   SLANet_plus_onnx / SLANeXt_*_onnx
      │
      └── [可选] 表格单元格检测 ── 检测每个单元格位置
          RT-DETR-L_*_table_cell_det_onnx

用法:
  python ocr_inference.py                       # 对 ocr_test.png 做识别
  python ocr_inference.py my.jpg                # 对指定图片做识别
  python ocr_inference.py my.jpg --save-json    # 额外保存 JSON 结果

通过 config.json 配置每个功能的开关和模型选择。
"""

import os
import sys
import json
import csv
import traceback

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont
from paddlex import create_model


# ─── 配置加载 ────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "engine": "onnxruntime",
    "text_detection": {"enabled": True, "model_dir": "auto"},
    "text_recognition": {"enabled": True, "model_dir": "auto"},
    "textline_orientation": {"enabled": True, "model_dir": "auto"},
    "doc_orientation_classify": {"enabled": False, "model_dir": "auto"},
    "doc_unwarping": {"enabled": False, "model_dir": "auto"},
    "doc_layout": {"enabled": False, "model_dir": "auto"},
    "table_cls": {"enabled": False, "model_dir": "auto"},
    "table_structure": {"enabled": False, "model_dir": "auto"},
    "table_cell_det_wired": {"enabled": False, "model_dir": "auto"},
    "table_cell_det_wireless": {"enabled": False, "model_dir": "auto"},
    "font": {"path": "auto", "size": 18, "color": [255, 0, 0]},
    "draw_detection_boxes": True,
    "box_color": [0, 255, 0],
    "box_thickness": 2,
}


def load_config(config_path: str) -> dict:
    """加载 config.json，缺失字段用默认值填充"""
    cfg = dict(DEFAULT_CONFIG)

    def _deep_merge(base, override):
        for k, v in override.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                _deep_merge(base[k], v)
            else:
                base[k] = v

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        _deep_merge(cfg, loaded)
    return cfg


# ─── 模型自动发现 ────────────────────────────────────────────────────

# 每个功能的关键词匹配规则（用于自动发现目录）
_FEATURE_KEYWORDS = {
    "text_detection": ["det"],
    "text_recognition": ["rec"],
    "textline_orientation": ["textline_ori", "textline_orientation"],
    "doc_orientation_classify": ["doc_ori", "doc_orientation"],
    "doc_unwarping": ["uvdoc"],
    "doc_layout": ["doclayout", "doc_layout"],
    "table_cls": ["table_cls", "table_class"],
    "table_structure": ["slanet", "slanext"],
    "table_cell_det_wired": ["wired_table_cell_det"],
    "table_cell_det_wireless": ["wireless_table_cell_det"],
}


def _read_model_name_from_yml(model_dir: str) -> str:
    """从 inference.yml 读取 model_name"""
    yml_path = os.path.join(model_dir, "inference.yml")
    if os.path.exists(yml_path):
        try:
            with open(yml_path) as f:
                cfg = yaml.safe_load(f)
            return cfg.get("Global", {}).get("model_name", "")
        except Exception:
            pass
    name = os.path.basename(model_dir.rstrip("/\\"))
    for suffix in ["_onnx", "-onnx"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def _discover_models(base_dir: str) -> dict:
    """
    扫描 base_dir 下的子目录，按目录名关键词匹配功能。
    返回: { 'text_detection': '路径', ... }
    每个功能只返回匹配到的一个。
    """
    candidates = {feat: [] for feat in _FEATURE_KEYWORDS}
    if not os.path.isdir(base_dir):
        return {}

    for entry in os.listdir(base_dir):
        full = os.path.join(base_dir, entry)
        if not os.path.isdir(full):
            continue
        if not os.path.exists(os.path.join(full, "inference.onnx")):
            continue
        name_low = entry.lower()
        for feature, keywords in _FEATURE_KEYWORDS.items():
            if any(k in name_low for k in keywords):
                candidates[feature].append(full.replace("\\", "/"))

    result = {}
    for feat, items in candidates.items():
        if items:
            items.sort()
            result[feat] = items[0]
    return result


# ─── 安全取值辅助 ────────────────────────────────────────────────────

def _safe_get(obj: dict, *keys):
    """链式 .get，返回第一个非 None 的值，避免 numpy ambiguous truth"""
    for k in keys:
        v = obj.get(k, None)
        if v is not None:
            return v
    return None


# ─── 字体 ────────────────────────────────────────────────────────────

def _find_font(font_path: str = "auto", size: int = 20) -> ImageFont.FreeTypeFont:
    """查找系统可用的中文字体"""
    if font_path != "auto":
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:
            print(f"  [警告] 指定字体 '{font_path}' 不可用，自动搜索")

    candidates = [
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/yahei.ttf",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttf",
    ]
    for fp in candidates:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


# ─── 图像旋转辅助 ────────────────────────────────────────────────────

def _rotate_img(img: np.ndarray, angle: int) -> np.ndarray:
    """按角度(0/90/180/270)旋转图像"""
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


# ─── 表格输出辅助 ────────────────────────────────────────────────────

def _table_to_csv(table_data: dict, save_path: str):
    """将表格结构识别结果保存为 CSV"""
    rows = table_data.get("rows", [])
    cols = table_data.get("cols", [])
    cells = table_data.get("cells", [])

    # 支持 rows/cols 为整数（行/列数）或列表
    if isinstance(rows, (int, np.integer)):
        rows = list(range(int(rows)))
    if isinstance(cols, (int, np.integer)):
        cols = list(range(int(cols)))

    if not rows or not cols:
        return

    grid = [[""] * len(cols) for _ in range(len(rows))]
    for cell in cells:
        r1, r2 = cell.get("row_range", [0, 1])
        c1, c2 = cell.get("col_range", [0, 1])
        text = cell.get("text", "")
        grid[r1][c1] = text

    with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerows(grid)
    print(f"  [表格] CSV 已保存: {save_path}")


def _table_to_html(table_data: dict) -> str:
    """将表格结构识别结果转为 HTML"""
    rows = table_data.get("rows", [])
    cols = table_data.get("cols", [])
    cells = table_data.get("cells", [])

    # 支持 rows/cols 为整数（行/列数）或列表
    if isinstance(rows, (int, np.integer)):
        rows = list(range(int(rows)))
    if isinstance(cols, (int, np.integer)):
        cols = list(range(int(cols)))

    if not rows or not cols:
        return "<table></table>"

    grid = [[{"text": "", "rowspan": 1, "colspan": 1}
             for _ in range(len(cols))] for _ in range(len(rows))]

    for cell in cells:
        r1, r2 = cell.get("row_range", [0, 1])
        c1, c2 = cell.get("col_range", [0, 1])
        text = cell.get("text", "")
        grid[r1][c1] = {
            "text": text,
            "rowspan": r2 - r1,
            "colspan": c2 - c1,
        }

    html = ["<table border='1' style='border-collapse:collapse;'>"]
    for r in range(len(rows)):
        html.append("<tr>")
        for c in range(len(cols)):
            cell = grid[r][c]
            if cell is None:
                continue
            if cell["colspan"] > 1:
                for cc in range(c + 1, c + cell["colspan"]):
                    grid[r][cc] = None
            if cell["rowspan"] > 1:
                for rr in range(r + 1, r + cell["rowspan"]):
                    for cc in range(c, c + cell["colspan"]):
                        if rr < len(grid) and cc < len(grid[rr]):
                            grid[rr][cc] = None

            attrs = ""
            if cell["colspan"] > 1:
                attrs += f" colspan='{cell['colspan']}'"
            if cell["rowspan"] > 1:
                attrs += f" rowspan='{cell['rowspan']}'"
            html.append(f"  <td{attrs}>{cell['text']}</td>")
        html.append("</tr>")
    html.append("</table>")
    return "\n".join(html)


# ─── 绘制 ────────────────────────────────────────────────────────────

def draw_results(
    img: np.ndarray,
    results: list,
    cfg: dict,
    save_path: str = "ocr_result.png",
    table_results: list = None,
) -> np.ndarray:
    out = img.copy()
    font_cfg = cfg.get("font", {})
    font = _find_font(font_cfg.get("path", "auto"), font_cfg.get("size", 18))
    text_color = tuple(font_cfg.get("color", [255, 0, 0]))

    if cfg.get("draw_detection_boxes", True):
        box_color = tuple(cfg.get("box_color", [0, 255, 0]))
        thickness = cfg.get("box_thickness", 2)
        for r in results:
            if r.get("box") is not None:
                pts = r["box"].astype(np.int32)
                cv2.polylines(out, [pts], isClosed=True, color=box_color, thickness=thickness)

        if table_results:
            table_color = (255, 0, 0)
            for tr in table_results:
                if tr.get("box") is not None:
                    pts = tr["box"].astype(np.int32)
                    cv2.polylines(out, [pts], isClosed=True, color=table_color, thickness=thickness + 2)

    pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    for r in results:
        text = r.get("text", "")
        if not text:
            continue
        box = r.get("box")
        if box is None:
            continue
        pts = box.astype(np.int32)
        x = int(pts[0, 0])
        y = int(pts[0, 1]) - font_cfg.get("size", 16) - 2
        draw.text((x, max(y, 4)), text, font=font, fill=text_color)

    out = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, out)
    print(f"\n结果图保存: {save_path}")
    return out


# ─── 主 OCR 流水线 ──────────────────────────────────────────────────

class OCRPipeline:
    """完整的 OCR 流水线，按需加载模型"""

    def __init__(self, config: dict, base_dir: str):
        self.cfg = config
        self.base_dir = base_dir
        self.models = {}
        self._discovered = _discover_models(base_dir)

        for feat, d in sorted(self._discovered.items()):
            name = _read_model_name_from_yml(d)
            print(f"  [发现] {name:<35s} <- {d}")

    def _get_dir(self, feature: str):
        f_cfg = self.cfg.get(feature, {})
        if not f_cfg.get("enabled", True):
            return None

        model_dir = f_cfg.get("model_dir", "auto")
        if model_dir and model_dir != "auto":
            return model_dir.replace("\\", "/") if os.path.isdir(model_dir) else None

        return self._discovered.get(feature)

    def _load(self, feature: str):
        if feature in self.models:
            return self.models[feature]
        model_dir = self._get_dir(feature)
        if model_dir is None:
            self.models[feature] = None
            return None
        model_name = _read_model_name_from_yml(model_dir)
        if not model_name:
            print(f"  [错误] 无法从 {model_dir} 读取模型名称")
            self.models[feature] = None
            return None
        print(f"  [加载] {model_name} <- {model_dir}")
        try:
            model = create_model(
                model_name=model_name,
                model_dir=model_dir,
                engine=self.cfg.get("engine", "onnxruntime"),
            )
            self.models[feature] = model
            return model
        except Exception as e:
            print(f"  [错误] 加载模型 {model_name} 失败: {e}")
            self.models[feature] = None
            return None

    # ── 文档方向分类 ──

    def _step_doc_orientation(self, img: np.ndarray):
        model = self._load("doc_orientation_classify")
        if model is None:
            return img, 0

        print("  [方向分类] 运行中...")
        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            for r in model.predict(img_rgb):
                if isinstance(r, dict):
                    class_ids = r.get("class_ids", [0])
                    if len(class_ids) > 0:
                        raw = class_ids[0]
                        label = int(float(raw.item() if isinstance(raw, np.ndarray) else raw))
                    else:
                        label = 0
                elif isinstance(r, (list, np.ndarray)):
                    if len(r) > 0:
                        raw = r[0]
                        label = int(float(raw.item() if isinstance(raw, np.ndarray) else raw))
                    else:
                        label = 0
                else:
                    label = 0
                angle_map = {0: 0, 1: 90, 2: 180, 3: 270}
                angle = angle_map.get(label, 0)
                if angle != 0:
                    img = _rotate_img(img, angle)
                    print(f"  [方向分类] 旋转 {angle}°")
                else:
                    print(f"  [方向分类] 方向正确 (0°)")
                return img, angle
        except Exception as e:
            print(f"  [方向分类] 失败: {e}")
        return img, 0

    # ── 文档畸变校正 ──

    def _step_doc_unwarping(self, img: np.ndarray):
        model = self._load("doc_unwarping")
        if model is None:
            return img

        print("  [畸变校正] 运行中...")
        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            for r in model.predict(img_rgb):
                warped = r.get("warped", None)
                if warped is not None:
                    print("  [畸变校正] 完成")
                    return warped
                pred_img = r.get("img", None)
                if pred_img is not None:
                    print("  [畸变校正] 完成")
                    return pred_img
        except Exception as e:
            print(f"  [畸变校正] 失败: {e}")
        return img

    # ── 版面分析 ──

    def _step_doc_layout(self, img: np.ndarray):
        model = self._load("doc_layout")
        if model is None:
            return []

        print("  [版面分析] 运行中...")
        layout_results = []
        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            for r in model.predict(img_rgb):
                # PP-DocLayoutV3 输出: boxes 为 list[dict]
                boxes = _safe_get(r, "boxes", "dt_polys")
                if boxes is None:
                    break

                for i, item in enumerate(boxes):
                    if isinstance(item, dict):
                        # 新格式: 每个 box 是 dict，内含 label/score/polygon_points
                        label = _safe_get(item, "label", "class_name")
                        if label is None:
                            label = f"region_{i}"
                        raw_score = _safe_get(item, "score", "class_score")
                        score = float(raw_score) if raw_score is not None else 0.0
                        poly = _safe_get(item, "polygon_points", "coordinate", "box")
                        if poly is not None:
                            poly_arr = np.array(poly, dtype=np.float32)
                            # coordinate 格式为 [x1,y1,x2,y2]，需转为 (4,2)
                            if poly_arr.ndim == 1 and poly_arr.shape[0] == 4:
                                x1, y1, x2, y2 = poly_arr
                                poly_arr = np.array(
                                    [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                                    dtype=np.float32,
                                )
                            layout_results.append({
                                "box": poly_arr,
                                "label": label,
                                "score": score,
                            })
                    else:
                        # 旧格式兼容: 单独的 boxes/labels/scores 列表
                        label = ""
                        score = 0.0
                        _labels = r.get("labels", [])
                        if _labels is not None and i < len(_labels):
                            label = _labels[i]
                        _scores = r.get("scores", [])
                        if _scores is not None and i < len(_scores):
                            raw_score = _scores[i]
                            if isinstance(raw_score, dict):
                                score = float(raw_score.get("score", raw_score.get("value", 0.0)))
                            elif isinstance(raw_score, (int, float)):
                                score = float(raw_score)
                        layout_results.append({
                            "box": np.array(item, dtype=np.float32),
                            "label": label or "unknown",
                            "score": score,
                        })

                if layout_results:
                    print(f"  [版面分析] 发现 {len(layout_results)} 个区域")
                    for lr in layout_results:
                        print(f"    - {lr['label']} (score: {lr['score']:.3f})")
                break  # 只取第一张结果
        except Exception as e:
            print(f"  [版面分析] 失败: {e}")
            traceback.print_exc()
        return layout_results

    # ── 文字行方向分类 ──

    def _step_textline_orientation(self, img: np.ndarray, polys: list):
        model = self._load("textline_orientation")
        if model is None:
            return polys

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        adjusted = []
        for i, poly in enumerate(polys):
            xmin = max(0, int(poly[:, 0].min()) - 2)
            ymin = max(0, int(poly[:, 1].min()) - 2)
            xmax = min(img.shape[1], int(poly[:, 0].max()) + 2)
            ymax = min(img.shape[0], int(poly[:, 1].max()) + 2)
            crop = img_rgb[ymin:ymax, xmin:xmax]
            if crop.size == 0:
                adjusted.append(poly)
                continue
            try:
                for rr in model.predict(crop):
                    raw_ids = rr.get("class_ids", [0])
                    if isinstance(raw_ids, (list, np.ndarray)):
                        if len(raw_ids) > 0:
                            raw = raw_ids[0]
                            _ = int(float(raw.item() if isinstance(raw, np.ndarray) else raw))
                    else:
                        _ = int(float(raw_ids))
            except Exception:
                pass
            adjusted.append(poly)
        return adjusted

    # ── 表格类型分类 ──

    def _step_table_cls(self, img: np.ndarray, table_regions: list):
        model = self._load("table_cls")
        if model is None or not table_regions:
            for tr in table_regions:
                tr["type"] = tr.get("type", "unknown")
            return table_regions

        print("  [表格分类] 运行中...")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for tr in table_regions:
            box = tr["box"]
            xmin = max(0, int(box[:, 0].min()) - 4)
            ymin = max(0, int(box[:, 1].min()) - 4)
            xmax = min(img.shape[1], int(box[:, 0].max()) + 4)
            ymax = min(img.shape[0], int(box[:, 1].max()) + 4)
            crop = img_rgb[ymin:ymax, xmin:xmax]
            if crop.size == 0:
                continue
            try:
                for rr in model.predict(crop):
                    # 先转 Python 原生类型，避免 numpy ambiguity
                    raw_ids = rr.get("class_ids", [0])
                    if hasattr(raw_ids, "tolist"):
                        raw_ids = raw_ids.tolist()

                    # 展平嵌套列表（模型输出可能带 batch 维度）
                    while isinstance(raw_ids, (list, np.ndarray)) and len(raw_ids) == 1 \
                            and isinstance(raw_ids[0], (list, np.ndarray)):
                        raw_ids = raw_ids[0]

                    if isinstance(raw_ids, (list, np.ndarray)):
                        label_id = int(raw_ids[0]) if len(raw_ids) > 0 else 0
                    else:
                        label_id = int(raw_ids)

                    raw_scores = rr.get("scores")
                    if raw_scores is not None:
                        if hasattr(raw_scores, "tolist"):
                            raw_scores = raw_scores.tolist()
                        if isinstance(raw_scores, list) and len(raw_scores) > 0:
                            s = raw_scores[0]
                            if isinstance(s, dict):
                                score = float(s.get("score", s.get("value", 0.0)))
                            else:
                                score = float(s)
                        elif isinstance(raw_scores, dict):
                            score = float(raw_scores.get("score", raw_scores.get("value", 0.0)))
                        else:
                            score = float(raw_scores) if raw_scores else 0.0
                    else:
                        score = 0.0

                    print(f"    [表格] #{i + 1}: 类型未知且没有可用的单元格检测模型")
                    continue
                try:
                    _infer_cells(model, tr, i, "unknown")
                except Exception as e:
                    print(f"    [表格] #{i + 1}: 单元格检测失败: {e}")
                continue

            if model_key not in _cell_det_models:
                _cell_det_models[model_key] = self._load(model_key)
            model = _cell_det_models[model_key]
            if model is None:
                print(f"    [表格] #{i + 1}: {tbl_type} 模型未启用或未找到")
                continue

            try:
                _infer_cells(model, tr, i, tbl_type)
            except Exception as e:
                print(f"    [表格] #{i + 1}: 单元格检测失败: {e}")

        return table_regions

    # ── 运行流水线 ──

    def predict(self, image_path: str):
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"无法读取图片: {image_path}")

        print(f"\n输入图片尺寸: {img.shape[1]}x{img.shape[0]}")
        print("-" * 50)

        # 1. 文档方向分类
        img, rotate_angle = self._step_doc_orientation(img)
        if rotate_angle != 0:
            print(f"  [信息] 图片已旋转 {rotate_angle}°")
            print("-" * 50)

        # 2. 文档畸变校正
        img = self._step_doc_unwarping(img)
        print("-" * 50)

        # 3. 版面分析
        layout_results = self._step_doc_layout(img)
        table_regions = [lr for lr in layout_results if "table" in lr.get("label", "").lower()]
        if table_regions:
            print(f"  [版面] 发现 {len(table_regions)} 个表格区域")
        print("-" * 50)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 4. 文本检测
        det = self._load("text_detection")
        if det is None:
            print("  [检测] 未启用，跳过")
            return [], []

        print("  [检测] 运行中...")
        polys = None
        for r in det.predict(image_path):
            polys = r.get("dt_polys")
            break

        if polys is None or len(polys) == 0:
            print("  [检测] 未检测到文字框")
            return [], []

        print(f"  [检测] 发现 {len(polys)} 个文本框")
        print("-" * 50)

        # 5. 文字行方向分类
        polys = self._step_textline_orientation(img, polys)
        print("-" * 50)

        # 6. 文本识别
        rec = self._load("text_recognition")
        ocr_results = []

        for i, poly in enumerate(polys):
            xmin = max(0, int(poly[:, 0].min()) - 2)
            ymin = max(0, int(poly[:, 1].min()) - 2)
            xmax = min(img.shape[1], int(poly[:, 0].max()) + 2)
            ymax = min(img.shape[0], int(poly[:, 1].max()) + 2)
            crop = img_rgb[ymin:ymax, xmin:xmax]

            if crop.size == 0:
                continue

            text, score = "", 0.0
            if rec is not None:
                for rr in rec.predict(crop):
                    text = rr.get("rec_text", "")
                    score = rr.get("rec_score", 0.0)
                    break

            ocr_results.append({
                "box": np.array(poly, dtype=np.float32),
                "text": text,
                "score": score,
            })
            if text:
                print(f"  [{i + 1:3d}] {text}  (可信度: {score:.4f})")
            else:
                print(f"  [{i + 1:3d}] (空)")

        if ocr_results:
            print("-" * 50)

        # 7. 表格类型分类
        if table_regions:
            table_regions = self._step_table_cls(img, table_regions)
            print("-" * 50)

        # 8. 表格结构识别
        if table_regions:
            table_regions = self._step_table_structure(img, table_regions)
            print("-" * 50)

        # 9. 表格单元格检测
        if table_regions:
            table_regions = self._step_table_cell_det(img, table_regions)
            print("-" * 50)

        return ocr_results, table_regions


# ─── 命令行入口 ──────────────────────────────────────────────────────

def main():
    base = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base, "config.json")
    cfg = load_config(config_path)

    print("=" * 60)
    print("  OCR 推理 — 纯本地 · 不联网")
    print(f"  配置: {config_path}")
    print("=" * 60)

    save_json = "--save-json" in sys.argv

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        image_path = args[0]
    else:
        image_path = os.path.join(base, "ocr_test.png")
        if not os.path.exists(image_path):
            print(f"错误: 找不到默认图片 {image_path}")
            print("用法: python ocr_inference.py <图片路径> [--save-json]")
            print("  或: 将图片命名为 ocr_test.png 放在当前目录")
            sys.exit(1)

    print(f"输入: {image_path}")
    print()

    pipeline = OCRPipeline(cfg, base)
    ocr_results, table_results = pipeline.predict(image_path)

    if not ocr_results and not table_results:
        print("\n  (未识别到任何内容)")
        return

    print("\n" + "=" * 60)
    print("  处理完成!")
    print(f"  文本块: {len(ocr_results)}")
    if table_results:
        print(f"  表格: {len(table_results)}")
    print("=" * 60)

    img = cv2.imread(image_path)
    draw_results(img, ocr_results, cfg, os.path.join(base, "ocr_result.png"), table_results)

    if save_json:
        json_path = os.path.join(base, "ocr_result.json")
        output = {
            "text_blocks": [
                {
                    "box": r["box"].tolist() if isinstance(r["box"], np.ndarray) else r["box"],
                    "text": r["text"],
                    "score": r["score"],
                }
                for r in ocr_results
            ],
            "tables": [
                {
                    "box": tr.get("box", []).tolist() if isinstance(tr.get("box"), np.ndarray) else tr.get("box"),
                    "type": tr.get("type", "unknown"),
                    "cls_score": tr.get("cls_score", None),
                    "rows": tr.get("rows", 0),
                    "cols": tr.get("cols", 0),
                    "cells": tr.get("cells", []),
                    "cell_boxes": [
                        cb.tolist() if isinstance(cb, np.ndarray) else cb
                        for cb in tr.get("cell_boxes", [])
                    ],
                    "html": tr.get("html", ""),
                    "csv_path": tr.get("csv_path", ""),
                }
                for tr in table_results
            ],
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\nJSON 结果保存: {json_path}")


if __name__ == "__main__":
    main()

                    tr["type"] = "wired" if label_id == 0 else "wireless"
                    tr["cls_score"] = score
                    print(f"    [表格] 类型={tr['type']}, score={score:.3f}")
            except Exception as e:
                print(f"    [表格分类] 失败: {e}")
                import traceback
                traceback.print_exc()
                tr["type"] = "unknown"
        return table_regions

    # ── 表格结构识别 ──

    def _step_table_structure(self, img: np.ndarray, table_regions: list):
        model = self._load("table_structure")
        if model is None or not table_regions:
            return table_regions

        print("  [表格结构] 运行中...")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        base_dir = os.path.dirname(os.path.abspath(__file__))

        for i, tr in enumerate(table_regions):
            box = tr["box"]
            xmin = max(0, int(box[:, 0].min()) - 8)
            ymin = max(0, int(box[:, 1].min()) - 8)
            xmax = min(img.shape[1], int(box[:, 0].max()) + 8)
            ymax = min(img.shape[0], int(box[:, 1].max()) + 8)
            crop = img_rgb[ymin:ymax, xmin:xmax]
            if crop.size == 0:
                continue

            try:
                for rr in model.predict(crop):
                    html_str = _safe_get(rr, "pred_html", "html")
                    cells_raw = rr.get("cells", [])

                    cells = []
                    rows_set = set()
                    cols_set = set()
                    for cell in cells_raw:
                        cells.append({
                            "row_range": cell.get("row_range", [0, 1]),
                            "col_range": cell.get("col_range", [0, 1]),
                            "text": cell.get("text", ""),
                        })
                        rows_set.add(cell.get("row_range", [0, 1])[0])
                        cols_set.add(cell.get("col_range", [0, 1])[0])

                    row_count = max(rows_set) + 1 if rows_set else 1
                    col_count = max(cols_set) + 1 if cols_set else 1

                    table_data = {
                        "html": html_str or "",
                        "cells": cells,
                        "rows": row_count,
                        "cols": col_count,
                    }

                    if html_str:
                        import re
                        row_matches = re.findall(r"<tr[^>]*>", html_str, re.I)
                        first_tr = re.search(r"<tr[^>]*>(.*?)</tr>", html_str, re.I | re.S)
                        if first_tr:
                            col_count = len(re.findall(r"<t[dh][^>]*>", first_tr.group(1), re.I))
                        row_count = len(row_matches)
                        table_data["rows"] = max(row_count, 1)
                        table_data["cols"] = max(col_count, 1)

                    csv_name = f"table_{i + 1}.csv"
                    csv_path = os.path.join(base_dir, csv_name)
                    _table_to_csv(table_data, csv_path)

                    if not html_str:
                        html_str = _table_to_html(table_data)
                    table_data["html"] = html_str

                    html_path = os.path.join(base_dir, f"table_{i + 1}.html")
                    with open(html_path, "w", encoding="utf-8") as f:
                        f.write(
                            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
                            f"<title>Table {i + 1}</title></head><body>"
                            f"{html_str}</body></html>"
                        )
                    print(f"  [表格] #{i + 1}: {row_count}行 x {col_count}列")
                    print(f"  [表格] HTML 已保存: {html_path}")

                    tr["html"] = html_str
                    tr["csv_path"] = csv_path
                    tr["html_path"] = html_path
                    tr["rows"] = row_count
                    tr["cols"] = col_count
                    tr["cells"] = cells
                    break
            except Exception as e:
                print(f"  [表格结构] 失败: {e}")
                traceback.print_exc()

        return table_regions

    # ── 表格单元格检测 ──

    def _step_table_cell_det(self, img: np.ndarray, table_regions: list):
        if not table_regions:
            return table_regions

        _cell_det_models = {}
        print("  [表格单元格] 运行中...")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        def _infer_cells(_model, _tr, _i, _label):
            _box = _tr["box"]
            _xmin = max(0, int(_box[:, 0].min()) - 4)
            _ymin = max(0, int(_box[:, 1].min()) - 4)
            _xmax = min(img.shape[1], int(_box[:, 0].max()) + 4)
            _ymax = min(img.shape[0], int(_box[:, 1].max()) + 4)
            _crop = img_rgb[_ymin:_ymax, _xmin:_xmax]
            if _crop.size == 0:
                return
            for _rr in _model.predict(_crop):
                _cell_boxes = _safe_get(_rr, "dt_polys", "boxes")
                if _cell_boxes is not None:
                    _parsed = []
                    for cb in _cell_boxes:
                        if isinstance(cb, dict):
                            # boxes 可能是 [{coordinate: [[x,y],...]}, ...]
                            _coord = cb.get("coordinate", None)
                            if _coord is not None:
                                _parsed.append(np.array(_coord, dtype=np.float32))
                        elif isinstance(cb, (list, tuple)):
                            _parsed.append(np.array(cb, dtype=np.float32))
                        elif isinstance(cb, np.ndarray):
                            _parsed.append(cb.astype(np.float32))
                    _tr["cell_boxes"] = _parsed
                    print(f"    [表格] #{_i + 1} ({_label}): {len(_parsed)} 个单元格")
                break

        for i, tr in enumerate(table_regions):
            tbl_type = tr.get("type", "unknown")
            if tbl_type == "wired":
                model_key = "table_cell_det_wired"
            elif tbl_type == "wireless":
                model_key = "table_cell_det_wireless"
            else:
                model = self._load("table_cell_det_wired")
                if model is None:
                    model = self._load("table_cell_det_wireless")
                if model is None:
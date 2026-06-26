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
PP-OCR 同系列模型下载工具
========================
只下载本地缺失的 ONNX 模型（全部用最大版本）。
所有模型从 HuggingFace PaddlePaddle 官方仓库下载。

用法:
  python download_models.py                    # 下载所有缺失模型
  python download_models.py --list             # 只列出模型清单
  python download_models.py --only <module>    # 只下载指定模块
"""

import os
import sys
import json
import urllib.request
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── 模型清单 ──────────────────────────────────────────────────────────
# 所有模型都用 ONNX 格式，全部选最大版本
# HuggingFace repo 名称 → 本地目录名

MODELS = [
    # ── OCR 核心（已有，默认跳过） ──
    {
        "repo": "PaddlePaddle/PP-OCRv6_medium_det_onnx",
        "dir_name": "PP-OCRv6_medium_det_onnx",
        "description": "文本检测 (PP-OCRv6 medium)",
        "required": True,
    },
    {
        "repo": "PaddlePaddle/PP-OCRv6_medium_rec_onnx",
        "dir_name": "PP-OCRv6_medium_rec_onnx",
        "description": "文本识别 (PP-OCRv6 medium)",
        "required": True,
    },
    {
        "repo": "PaddlePaddle/PP-LCNet_x1_0_textline_ori_onnx",
        "dir_name": "PP-LCNet_x1_0_textline_ori_onnx",
        "description": "文字行方向分类 (PP-LCNet x1.0)",
        "required": True,
    },
    {
        "repo": "PaddlePaddle/UVDoc_onnx",
        "dir_name": "UVDoc_onnx",
        "description": "文档畸变校正/展平 (UVDoc)",
        "required": False,
    },
    # ── 文档方向分类（新增） ──
    {
        "repo": "PaddlePaddle/PP-LCNet_x1_0_doc_ori_onnx",
        "dir_name": "PP-LCNet_x1_0_doc_ori_onnx",
        "description": "文档方向分类 (PP-LCNet x1.0)",
        "required": False,
    },
    # ── 文档版面分析（新增：检测表格、标题等区域） ──
    {
        "repo": "PaddlePaddle/PP-DocLayoutV3_onnx",
        "dir_name": "PP-DocLayoutV3_onnx",
        "description": "文档版面分析 (PP-DocLayoutV3)",
        "required": False,
    },
    # ── 表格类型分类（新增） ──
    {
        "repo": "PaddlePaddle/PP-LCNet_x1_0_table_cls_onnx",
        "dir_name": "PP-LCNet_x1_0_table_cls_onnx",
        "description": "表格类型分类：有线/无线 (PP-LCNet x1.0)",
        "required": False,
    },
    # ── 表格结构识别（新增） ──
    {
        "repo": "PaddlePaddle/SLANet_plus_onnx",
        "dir_name": "SLANet_plus_onnx",
        "description": "表格结构识别 (SLANet_plus)，通用型",
        "required": False,
    },
    # ── 表格单元格检测（新增） ──
    {
        "repo": "PaddlePaddle/RT-DETR-L_wired_table_cell_det_onnx",
        "dir_name": "RT-DETR-L_wired_table_cell_det_onnx",
        "description": "有线表格单元格检测 (RT-DETR-L)",
        "required": False,
    },
    {
        "repo": "PaddlePaddle/RT-DETR-L_wireless_table_cell_det_onnx",
        "dir_name": "RT-DETR-L_wireless_table_cell_det_onnx",
        "description": "无线表格单元格检测 (RT-DETR-L)",
        "required": False,
    },
]


def download_hf_file(file_url: str, local_path: str, desc: str = "") -> bool:
    """下载单个文件，带进度显示"""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # 检查已有文件
    if os.path.exists(local_path):
        try:
            req_head = urllib.request.Request(file_url, method="HEAD",
                                              headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req_head, timeout=15) as resp:
                remote_size = int(resp.headers.get("Content-Length", 0))
            local_size = os.path.getsize(local_path)
            if local_size == remote_size:
                return True  # 已存在且完整
        except Exception:
            pass

    try:
        req = urllib.request.Request(file_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            with open(local_path, "wb") as f:
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded * 100 // total
                        print(f"\r  {desc} {downloaded//1024}KB/{total//1024}KB ({pct}%)", end="")
                    else:
                        print(f"\r  {desc} {downloaded//1024}KB", end="")
            print()
        return True
    except Exception as e:
        print(f"  [失败] {desc}: {e}")
        if os.path.exists(local_path):
            os.remove(local_path)
        return False


def download_repo(repo_id: str, target_dir: str, desc: str = "") -> bool:
    """从 HuggingFace 下载整个 repo 的所有文件"""
    # 查询 repo 文件列表
    api_url = f"https://huggingface.co/api/models/{repo_id}"
    try:
        req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            info = json.load(resp)
    except Exception as e:
        print(f"  [错误] 无法查询 {repo_id}: {e}")
        return False

    siblings = info.get("siblings", [])
    files = [s["rfilename"] for s in siblings
             if not s["rfilename"].startswith(".") and s.get("type") != "directory"]

    if not files:
        print(f"  [错误] {repo_id} 无可用文件")
        return False

    os.makedirs(target_dir, exist_ok=True)
    base_url = f"https://huggingface.co/{repo_id}/resolve/main"
    success = 0

    for filename in files:
        file_url = f"{base_url}/{filename}"
        local_path = os.path.join(target_dir, filename)
        fdesc = f"{desc} / {os.path.basename(filename)}"
        if download_hf_file(file_url, local_path, fdesc):
            success += 1

    ok = success == len(files)
    print(f"  {'✅ 完成' if ok else '⚠️ 部分完成'}: {success}/{len(files)} 文件")
    return ok


def main():
    print("=" * 60)
    print("  PP-OCR 同系列 ONNX 模型下载")
    print("  从 HuggingFace PaddlePaddle 官方仓库")
    print("=" * 60)
    print()

    # 处理参数
    only_module = None
    if "--list" in sys.argv:
        print("可用模型:")
        for m in MODELS:
            flag = "📌 必需" if m["required"] else "   可选"
            status = "✅" if os.path.isdir(os.path.join(BASE_DIR, m["dir_name"])) else "⬜"
            print(f"  {status} {flag} {m['description']:<30s} -> {m['dir_name']}")
        return

    if "--only" in sys.argv:
        idx = sys.argv.index("--only") + 1
        if idx < len(sys.argv):
            only_module = sys.argv[idx]

    # 逐个下载
    for model in MODELS:
        dir_name = model["dir_name"]
        target_path = os.path.join(BASE_DIR, dir_name)

        # 如果指定了 --only，只下载匹配的
        if only_module and only_module.lower() not in dir_name.lower() and only_module.lower() not in model["description"].lower():
            continue

        # 检查是否已完整
        has_onnx = os.path.exists(os.path.join(target_path, "inference.onnx"))
        has_yml = os.path.exists(os.path.join(target_path, "inference.yml"))
        if has_onnx and has_yml:
            print(f"[{model['description']}] ✅ 已存在: {target_path}")
            print()
            continue

        print(f"[{model['description']}] 📥 下载中...")
        ok = download_repo(model["repo"], target_path, model["description"])
        if ok:
            print(f"  ✅ 下载成功")
        else:
            print(f"  ❌ 下载失败，可稍后重试")
        print()

    print("=" * 60)
    print("  下载完成!")
    print("  运行 python ocr_inference.py <图片> 使用完整工作流")
    print("=" * 60)


if __name__ == "__main__":
    main()
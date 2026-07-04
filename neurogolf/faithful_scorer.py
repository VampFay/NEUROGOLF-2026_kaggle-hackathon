"""neurogolf/faithful_scorer.py — Faithful scoring using onnx-tool (the grader's tool)."""
from __future__ import annotations
import io, math, os, zipfile
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto

try:
    import onnx_tool
    _HAS_ONNX_TOOL = True
except ImportError:
    _HAS_ONNX_TOOL = False

from .constants import BANNED_OPS, INPUT_NAME, OUTPUT_NAME, IO_SHAPE, MAX_GRID, MAX_FILE_BYTES, NUM_COLORS


def compute_cost_via_onnx_tool(model):
    if not _HAS_ONNX_TOOL:
        return {"params": 0, "memory_bytes": 0, "cost": 0, "score": 1.0}
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            onnx_tool.model_profile(model, mcfg={'verbose': False})
    except Exception:
        return {"params": 0, "memory_bytes": 0, "cost": 0, "score": 1.0}
    output = buf.getvalue()
    for line in output.strip().split('\n'):
        if line.startswith('Total'):
            parts = line.split()
            nums = []
            for p in parts[1:]:
                cleaned = p.replace(',', '').replace('.', '').replace('-', '')
                if cleaned.isdigit():
                    nums.append(int(p.replace(',', '')))
            if len(nums) >= 3:
                macs, memory, params = nums[0], nums[1], nums[2]
                cost = params + memory
                score = max(1.0, 25.0 - math.log(cost)) if cost > 0 else 1.0
                return {"params": params, "memory_bytes": memory, "macs": macs,
                        "cost": cost, "score": score}
    return {"params": 0, "memory_bytes": 0, "cost": 0, "score": 1.0}


def compute_cost(model):
    if _HAS_ONNX_TOOL:
        result = compute_cost_via_onnx_tool(model)
        result["file_bytes"] = len(model.SerializeToString())
        return result
    # Fallback
    params = sum(int(np.prod(init.dims)) if init.dims else 1 for init in model.graph.initializer)
    file_bytes = len(model.SerializeToString())
    cost = params + file_bytes
    score = max(1.0, 25.0 - math.log(cost)) if cost > 0 else 1.0
    return {"params": params, "memory_bytes": 0, "file_bytes": file_bytes, "cost": cost, "score": score}


def validate_submission(zip_path):
    results = {}
    n_ok = 0
    n_fail = 0
    total_score = 0.0
    with zipfile.ZipFile(zip_path) as zf:
        for name in sorted(zf.namelist()):
            if not name.endswith(".onnx"): continue
            try:
                model_bytes = zf.read(name)
                model = onnx.load(io.BytesIO(model_bytes))
                ci = compute_cost(model)
                results[name] = ci
                if ci.get("cost", 0) > 0:
                    n_ok += 1
                    total_score += ci.get("score", 0)
                else:
                    n_fail += 1
            except Exception as e:
                results[name] = {"error": str(e)}
                n_fail += 1
    return {"n_files": len(results), "n_ok": n_ok, "n_fail": n_fail,
            "total_score": total_score, "results": results}

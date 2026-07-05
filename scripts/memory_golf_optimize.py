"""Memory golf optimization for expensive ONNX files.
Apply: int8 quantize Conv weights, drop Conv attributes, shorten tensor names.
"""
import sys, os, json, zipfile, math, re
sys.path.insert(0, "/home/z/my-project")
import numpy as np
import onnx
import onnx.helper as h
from onnx import TensorProto
import onnxruntime as ort
from neurogolf import arc_data, validator, faithful_scorer
from neurogolf.arc_data import grid_to_onehot, onehot_to_grid, get_pairs

ZIP_PATH = "/home/z/my-project/download/submission.zip"

def strip_metadata(model):
    model.ClearField("producer_name")
    model.ClearField("producer_version")
    model.ClearField("doc_string")
    model.ClearField("domain")
    model.ClearField("model_version")
    model.graph.ClearField("doc_string")
    if len(model.graph.name) > 1:
        model.graph.name = "g"
    return model

def drop_conv_attributes(model):
    """Drop default Conv attributes (strides, dilations, group, pads when default)."""
    for node in model.graph.node:
        if node.op_type == "Conv":
            # Check and remove default attributes
            attrs_to_remove = []
            for attr in node.attribute:
                if attr.name == "strides" and all(v == 1 for v in attr.ints):
                    attrs_to_remove.append(attr.name)
                elif attr.name == "dilations" and all(v == 1 for v in attr.ints):
                    attrs_to_remove.append(attr.name)
                elif attr.name == "group" and attr.i == 1:
                    attrs_to_remove.append(attr.name)
                elif attr.name == "pads" and all(v == 0 for v in attr.ints):
                    attrs_to_remove.append(attr.name)
                elif attr.name == "kernel_shape":
                    attrs_to_remove.append(attr.name)  # ORT infers from weight
            for attr_name in attrs_to_remove:
                for i, attr in enumerate(node.attribute):
                    if attr.name == attr_name:
                        del node.attribute[i]
                        break
    return model

def shorten_tensor_names(model):
    """Shorten tensor names to single characters."""
    name_map = {}
    counter = 0
    reserved = {"input", "output"}
    for init in model.graph.initializer:
        if init.name not in reserved and init.name not in name_map:
            name_map[init.name] = f"i{counter}"
            counter += 1
    for node in model.graph.node:
        for i, out in enumerate(node.output):
            if out not in reserved and out not in name_map and out:
                name_map[out] = f"n{counter}"
                counter += 1
    for init in model.graph.initializer:
        if init.name in name_map:
            init.name = name_map[init.name]
    for node in model.graph.node:
        for i, inp in enumerate(node.input):
            if inp in name_map:
                node.input[i] = name_map[inp]
        for i, out in enumerate(node.output):
            if out in name_map:
                node.output[i] = name_map[out]
    return model

def try_int8_quantize(model):
    """Try int8 quantize Conv weights. Only helps if there are Conv layers with float32 weights."""
    # Find Conv nodes with float32 weight initializers
    conv_nodes = [n for n in model.graph.node if n.op_type == "Conv"]
    if not conv_nodes:
        return model  # No Conv to quantize
    
    # For each Conv, try to quantize the weight
    for conv_node in conv_nodes:
        if len(conv_node.input) < 2:
            continue
        weight_name = conv_node.input[1]
        # Find the initializer
        init = None
        for i in model.graph.initializer:
            if i.name == weight_name:
                init = i
                break
        if init is None:
            continue
        # Check if it's float32
        if init.data_type != TensorProto.FLOAT:
            continue
        # Only quantize if the weight has enough elements (>30 for break-even)
        weight_size = 1
        for d in init.dims:
            weight_size *= d
        if weight_size < 30:
            continue  # Not worth it
        
        # Extract the weight data
        weight_array = onnx.numpy_helper.to_array(init)
        
        # Build quantization: scale = max(abs(weight)) / 127, zero_point = 0
        scale_val = float(np.max(np.abs(weight_array))) / 127.0
        if scale_val == 0:
            continue
        # Quantize to int8
        quantized = np.round(weight_array / scale_val).astype(np.int8)
        
        # Create new nodes: QuantizeLinear → QLinearConv → DequantizeLinear
        # Actually, replacing nodes in-place is complex. Skip for now if too risky.
        # Only do it for simple Conv (1 weight, optional bias)
        # 
        # For safety, skip int8 for now — it's risky and the gain is small
        # since most cost is from intermediate tensor memory, not weight params
        pass
    
    return model

def optimize_model(model, task):
    """Apply all optimizations and verify correctness."""
    original_bytes = model.SerializeToString()
    original_ci = faithful_scorer.compute_cost(model)
    original_cost = original_ci.get("cost", 0)
    
    # Apply optimizations
    model = strip_metadata(model)
    model = drop_conv_attributes(model)
    model = shorten_tensor_names(model)
    
    # Verify correctness after optimization
    try:
        e = validator.evaluate_model(model, task)
        if not e["eligible_for_points"]:
            # Revert
            return onnx.load_model_from_string(original_bytes), original_cost, False
    except Exception:
        return onnx.load_model_from_string(original_bytes), original_cost, False
    
    new_ci = faithful_scorer.compute_cost(model)
    new_cost = new_ci.get("cost", 0)
    return model, new_cost, True

def main():
    print("=== Memory Golf Optimization ===")
    
    # Read all files from submission.zip
    with zipfile.ZipFile(ZIP_PATH) as zf:
        files = {name: zf.read(name) for name in zf.namelist() if name.endswith(".onnx")}
    
    print(f"Found {len(files)} ONNX files")
    
    # Identify and optimize expensive files
    results = []
    total_original_score = 0
    total_optimized_score = 0
    
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf_out:
        for name in sorted(files.keys()):
            fbytes = files[name]
            model = onnx.load_model_from_string(fbytes)
            ci = faithful_scorer.compute_cost(model)
            cost = ci.get("cost", 0)
            score = max(1.0, 25.0 - math.log(cost)) if cost > 0 else 1.0
            total_original_score += score
            
            tid = int(name.replace("task", "").replace(".onnx", ""))
            try:
                task = arc_data.load_task(tid)
            except Exception:
                zf_out.writestr(name, fbytes)
                total_optimized_score += score
                continue
            
            if cost > 10000:
                # Optimize
                opt_model, opt_cost, success = optimize_model(model, task)
                if success and opt_cost < cost:
                    opt_ci = faithful_scorer.compute_cost(opt_model)
                    opt_score = opt_ci.get("score", 1.0)
                    gain = opt_score - score
                    print(f"  [OPT] {name}: cost {cost}→{opt_cost}, score {score:.2f}→{opt_score:.2f} ({'+' if gain>=0 else ''}{gain:.2f})")
                    zf_out.writestr(name, opt_model.SerializeToString())
                    total_optimized_score += opt_score
                    results.append({"name": name, "original_cost": cost, "optimized_cost": opt_cost,
                                    "original_score": score, "optimized_score": opt_score, "optimized": True})
                else:
                    zf_out.writestr(name, fbytes)
                    total_optimized_score += score
                    results.append({"name": name, "original_cost": cost, "optimized": False})
            else:
                # Already cheap, just write
                zf_out.writestr(name, fbytes)
                total_optimized_score += score
    
    print(f"\n=== Optimization Summary ===")
    print(f"Total original score: {total_original_score:.2f}")
    print(f"Total optimized score: {total_optimized_score:.2f}")
    print(f"Score gain: +{total_optimized_score - total_original_score:.2f}")
    opt_count = sum(1 for r in results if r.get("optimized"))
    print(f"Files optimized: {opt_count}/{len(results)}")

if __name__ == "__main__":
    main()

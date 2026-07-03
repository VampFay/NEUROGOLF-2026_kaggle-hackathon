"""
Empirical ONNX minimization research.
Builds many tiny ONNX graphs, serializes them, and measures byte sizes.
Also verifies each model runs in ONNX Runtime.
"""
import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np
import os, io, sys, json

OUT = "/home/z/my-project/data/onnx_research_out"
os.makedirs(OUT, exist_ok=True)

def sz(model):
    """Return serialized byte size of model."""
    return len(model.SerializeToString())

def save(model, name):
    p = os.path.join(OUT, name)
    onnx.save(model, p)
    return os.path.getsize(p)

def check(model):
    try:
        onnx.checker.check_model(model)
        return "ok"
    except Exception as e:
        return "CHECK_FAIL: " + str(e)[:80]

results = {}

# ---- 1. Bare minimum: identity ----
def make_identity(opset=13, name_doc=True, ir_version=None):
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, ["n"])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["n"])
    node = helper.make_node("Identity", ["X"], ["Y"])
    graph = helper.make_graph([node], "g", [X], [Y])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    if ir_version is not None:
        m.ir_version = ir_version
    if not name_doc:
        m.producer_name = ""
        m.producer_version = ""
        m.domain = ""
        m.model_version = 0
        m.doc_string = ""
        graph.name = ""
        graph.doc_string = ""
        for n in graph.node:
            n.name = ""
    return m

for ops in [7, 9, 11, 13, 15, 17, 19, 21]:
    try:
        m = make_identity(opset=ops)
        results[f"identity_opset{ops}"] = sz(m)
    except Exception as e:
        results[f"identity_opset{ops}"] = f"fail:{e}"

m = make_identity(opset=13, name_doc=False)
results["identity_opset13_stripped"] = sz(m)
for ir in [3, 4, 5, 6, 7, 8, 9]:
    try:
        m = make_identity(opset=13, name_doc=False, ir_version=ir)
        onnx.checker.check_model(m)
        results[f"identity_stripped_ir{ir}_size"] = sz(m)
        results[f"identity_stripped_ir{ir}_check"] = "ok"
    except Exception as e:
        results[f"identity_stripped_ir{ir}_check"] = "fail: " + str(e)[:60]

# ---- 2. Conv tests ----
def make_conv(name_len, weight_dtype=TensorProto.FLOAT, opset=13, use_initializer=True, n_in=1, n_out=1, kh=1, kw=1, with_bias=True):
    wname = "w" * name_len
    xname = "x" * name_len
    yname = "y" * name_len
    X = helper.make_tensor_value_info(xname, TensorProto.FLOAT, ["N", n_in, "h", "w"])
    Y = helper.make_tensor_value_info(yname, TensorProto.FLOAT, ["N", n_out, "h", "w"])
    if weight_dtype == TensorProto.FLOAT:
        w = np.random.randn(n_out, n_in, kh, kw).astype(np.float32)
        b = np.zeros(n_out, dtype=np.float32)
    wv = helper.make_tensor(wname, weight_dtype, list(w.shape), w.tobytes(), raw=True)
    bv = helper.make_tensor("b" * name_len, TensorProto.FLOAT, [n_out], b.tobytes(), raw=True)
    inputs = [xname, wname] + (["b" * name_len] if with_bias else [])
    node = helper.make_node("Conv", inputs, [yname], name="")
    graph = helper.make_graph([node], "", [X], [Y])
    if use_initializer:
        inits = [wv] + ([bv] if with_bias else [])
        graph.initializer.extend(inits)
    else:
        cw = helper.make_node("Constant", [], [wname], value=wv, name="")
        nodes = [cw]
        if with_bias:
            cb = helper.make_node("Constant", [], ["b" * name_len], value=bv, name="")
            nodes.append(cb)
        nodes.append(node)
        graph = helper.make_graph(nodes, "", [X], [Y])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    m.producer_name = ""; m.producer_version = ""; m.domain = ""
    m.model_version = 0; m.doc_string = ""
    return m

for nlen in [1, 4, 16, 64]:
    m = make_conv(name_len=nlen, use_initializer=True, n_in=10, n_out=10, kh=1, kw=1)
    results[f"conv1x1_100w_init_nlen{nlen}"] = sz(m)
m_const = make_conv(name_len=1, use_initializer=False, n_in=10, n_out=10, kh=1, kw=1)
results["conv1x1_100w_constant_nlen1"] = sz(m_const)
m_nobias = make_conv(name_len=1, use_initializer=True, n_in=10, n_out=10, kh=1, kw=1, with_bias=False)
results["conv1x1_100w_nobias_nlen1"] = sz(m_nobias)

m = make_conv(name_len=1, n_in=10, n_out=10, kh=3, kw=3, use_initializer=True)
results["conv3x3_900w_init_nlen1"] = sz(m)
m = make_conv(name_len=1, n_in=10, n_out=10, kh=3, kw=3, use_initializer=False)
results["conv3x3_900w_constant_nlen1"] = sz(m)

# Opset comparison for same conv
for ops in [7, 9, 11, 13, 15, 17, 19, 21]:
    try:
        m = make_conv(name_len=1, opset=ops, n_in=10, n_out=10, kh=1, kw=1)
        results[f"conv1x1_100w_opset{ops}"] = sz(m)
    except Exception as e:
        results[f"conv1x1_100w_opset{ops}"] = f"fail:{e}"

# ---- 3. int8 quantized conv (QLinearConv) ----
def make_qconv(opset=13, n_in=10, n_out=10, kh=1, kw=1):
    X = helper.make_tensor_value_info("x", TensorProto.UINT8, ["N", n_in, "h", "w"])
    Y = helper.make_tensor_value_info("y", TensorProto.UINT8, ["N", n_out, "h", "w"])
    w = np.random.randint(-127, 127, size=(n_out, n_in, kh, kw), dtype=np.int8)
    b = np.zeros(n_out, dtype=np.int32)
    wv = helper.make_tensor("w", TensorProto.INT8, list(w.shape), w.tobytes(), raw=True)
    bv = helper.make_tensor("b", TensorProto.INT32, [n_out], b.tobytes(), raw=True)
    x_scale = helper.make_tensor("xs", TensorProto.FLOAT, [], np.array(0.01, dtype=np.float32).tobytes(), raw=True)
    x_zp = helper.make_tensor("xz", TensorProto.UINT8, [], np.array(0, dtype=np.uint8).tobytes(), raw=True)
    w_scale = helper.make_tensor("ws", TensorProto.FLOAT, [], np.array(0.01, dtype=np.float32).tobytes(), raw=True)
    w_zp = helper.make_tensor("wz", TensorProto.INT8, [], np.array(0, dtype=np.int8).tobytes(), raw=True)
    y_scale = helper.make_tensor("ys", TensorProto.FLOAT, [], np.array(0.01, dtype=np.float32).tobytes(), raw=True)
    y_zp = helper.make_tensor("yz", TensorProto.UINT8, [], np.array(0, dtype=np.uint8).tobytes(), raw=True)
    inputs = ["x", "xs", "xz", "w", "ws", "wz", "b", "ys", "yz"]
    node = helper.make_node("QLinearConv", inputs, ["y"], name="")
    graph = helper.make_graph([node], "", [X], [Y])
    graph.initializer.extend([wv, bv, x_scale, x_zp, w_scale, w_zp, y_scale, y_zp])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    m.producer_name = ""; m.producer_version = ""; m.domain = ""
    m.model_version = 0; m.doc_string = ""
    return m

m = make_qconv(n_in=10, n_out=10, kh=1, kw=1)
results["qconv1x1_100w_int8"] = sz(m)
results["qconv1x1_100w_int8_check"] = check(m)
m3 = make_qconv(n_in=10, n_out=10, kh=3, kw=3)
results["qconv3x3_900w_int8"] = sz(m3)
results["qconv3x3_900w_int8_check"] = check(m3)

# QLinearConv without bias
def make_qconv_nobias(opset=13, n_in=10, n_out=10, kh=1, kw=1):
    X = helper.make_tensor_value_info("x", TensorProto.UINT8, ["N", n_in, "h", "w"])
    Y = helper.make_tensor_value_info("y", TensorProto.UINT8, ["N", n_out, "h", "w"])
    w = np.random.randint(-127, 127, size=(n_out, n_in, kh, kw), dtype=np.int8)
    wv = helper.make_tensor("w", TensorProto.INT8, list(w.shape), w.tobytes(), raw=True)
    x_scale = helper.make_tensor("xs", TensorProto.FLOAT, [], np.array(0.01, dtype=np.float32).tobytes(), raw=True)
    x_zp = helper.make_tensor("xz", TensorProto.UINT8, [], np.array(0, dtype=np.uint8).tobytes(), raw=True)
    w_scale = helper.make_tensor("ws", TensorProto.FLOAT, [], np.array(0.01, dtype=np.float32).tobytes(), raw=True)
    w_zp = helper.make_tensor("wz", TensorProto.INT8, [], np.array(0, dtype=np.int8).tobytes(), raw=True)
    y_scale = helper.make_tensor("ys", TensorProto.FLOAT, [], np.array(0.01, dtype=np.float32).tobytes(), raw=True)
    y_zp = helper.make_tensor("yz", TensorProto.UINT8, [], np.array(0, dtype=np.uint8).tobytes(), raw=True)
    inputs = ["x", "xs", "xz", "w", "ws", "wz", "ys", "yz"]
    node = helper.make_node("QLinearConv", inputs, ["y"], name="")
    graph = helper.make_graph([node], "", [X], [Y])
    graph.initializer.extend([wv, x_scale, x_zp, w_scale, w_zp, y_scale, y_zp])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    m.producer_name = ""; m.producer_version = ""; m.domain = ""
    m.model_version = 0; m.doc_string = ""
    return m

m = make_qconv_nobias(n_in=10, n_out=10, kh=1, kw=1)
results["qconv1x1_100w_int8_nobias"] = sz(m)
results["qconv1x1_100w_int8_nobias_check"] = check(m)

# ---- 4. Slice + Concat ----
def make_slice_concat(opset=13):
    X = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 10])
    Y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", 10])
    starts = helper.make_tensor("s", TensorProto.INT64, [1], np.array([0], dtype=np.int64).tobytes(), raw=True)
    ends = helper.make_tensor("e", TensorProto.INT64, [1], np.array([5], dtype=np.int64).tobytes(), raw=True)
    axes = helper.make_tensor("a", TensorProto.INT64, [1], np.array([1], dtype=np.int64).tobytes(), raw=True)
    sl = helper.make_node("Slice", ["x", "s", "e", "a"], ["p"], name="")
    cat = helper.make_node("Concat", ["p", "p"], ["y"], axis=1, name="")
    graph = helper.make_graph([sl, cat], "", [X], [Y])
    graph.initializer.extend([starts, ends, axes])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    m.producer_name = ""; m.producer_version = ""; m.domain = ""
    m.model_version = 0; m.doc_string = ""
    return m

def make_slice_attr(opset=10):
    X = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 10])
    Y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", 5])
    sl = helper.make_node("Slice", ["x"], ["p"], starts=[0], ends=[5], axes=[1], name="")
    graph = helper.make_graph([sl], "", [X], [Y])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    m.producer_name = ""; m.producer_version = ""; m.domain = ""
    m.model_version = 0; m.doc_string = ""
    return m

m = make_slice_concat(opset=13)
results["slice_concat_opset13"] = sz(m)
results["slice_concat_opset13_check"] = check(m)
m = make_slice_attr(opset=10)
results["slice_attr_opset10"] = sz(m)
results["slice_attr_opset10_check"] = check(m)

# ---- 5. Transpose ----
def make_transpose(opset=13):
    X = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", "C", "H", "W"])
    Y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", "H", "W", "C"])
    node = helper.make_node("Transpose", ["x"], ["y"], perm=[0,2,3,1], name="")
    graph = helper.make_graph([node], "", [X], [Y])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    m.producer_name=""; m.producer_version=""; m.domain=""
    m.model_version=0; m.doc_string=""
    return m

m = make_transpose(opset=13)
results["transpose_opset13"] = sz(m)
results["transpose_opset13_check"] = check(m)

# ---- 6. Resize nearest neighbor ----
def make_resize(opset=13):
    X = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 1, "H", "W"])
    Y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", 1, "H2", "W2"])
    roi = helper.make_tensor("r", TensorProto.FLOAT, [0], b"", raw=True)
    scales = helper.make_tensor("s", TensorProto.FLOAT, [4], np.array([1,1,2,2],dtype=np.float32).tobytes(), raw=True)
    node = helper.make_node("Resize", ["x","r","s"], ["y"], mode="nearest", name="")
    graph = helper.make_graph([node], "", [X], [Y])
    graph.initializer.extend([roi, scales])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    m.producer_name=""; m.producer_version=""; m.domain=""
    m.model_version=0; m.doc_string=""
    return m

m = make_resize(opset=13)
results["resize_nn_opset13"] = sz(m)
results["resize_nn_opset13_check"] = check(m)

# Resize opset 11 (uses scales as input, no roi input - actually 11 takes roi+scales, 13+ takes roi+scales/sizes)
# Actually opset 11: inputs [X, scales] or [X, roi, scales]
def make_resize_11(opset=11):
    X = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 1, "H", "W"])
    Y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", 1, "H2", "W2"])
    scales = helper.make_tensor("s", TensorProto.FLOAT, [4], np.array([1,1,2,2],dtype=np.float32).tobytes(), raw=True)
    node = helper.make_node("Resize", ["x","s"], ["y"], mode="nearest", name="")
    graph = helper.make_graph([node], "", [X], [Y])
    graph.initializer.extend([scales])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    m.producer_name=""; m.producer_version=""; m.domain=""
    m.model_version=0; m.doc_string=""
    return m

try:
    m = make_resize_11(opset=11)
    results["resize_nn_opset11"] = sz(m)
    results["resize_nn_opset11_check"] = check(m)
except Exception as e:
    results["resize_nn_opset11"] = f"fail: {e}"

# ---- 7. Color map ----
def make_colormap(n_colors=16, dtype=TensorProto.FLOAT, opset=13):
    X = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", n_colors, "H", "W"])
    Y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", 3, "H", "W"])
    w = np.random.randint(0, 255, size=(3, n_colors, 1, 1)).astype(np.float32)
    wv = helper.make_tensor("w", dtype, list(w.shape), w.tobytes(), raw=True)
    node = helper.make_node("Conv", ["x","w"], ["y"], name="")
    graph = helper.make_graph([node], "", [X], [Y])
    graph.initializer.extend([wv])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    m.producer_name=""; m.producer_version=""; m.domain=""
    m.model_version=0; m.doc_string=""
    return m

m = make_colormap(n_colors=16)
results["colormap_16colors_f32"] = sz(m)
results["colormap_16colors_f32_check"] = check(m)

# ---- 8. Constant folding test ----
def make_unfolded(opset=13):
    X = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["n"])
    Y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["n"])
    cval = helper.make_tensor("c", TensorProto.FLOAT, [3], np.array([1,2,3],dtype=np.float32).tobytes(), raw=True)
    cn = helper.make_node("Constant", [], ["c"], value=cval, name="")
    idn = helper.make_node("Identity", ["c"], ["c2"], name="")
    add = helper.make_node("Add", ["x","c2"], ["y"], name="")
    graph = helper.make_graph([cn, idn, add], "", [X], [Y])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    m.producer_name=""; m.producer_version=""; m.domain=""
    m.model_version=0; m.doc_string=""
    return m

def make_folded(opset=13):
    X = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["n"])
    Y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["n"])
    cval = helper.make_tensor("c", TensorProto.FLOAT, [3], np.array([1,2,3],dtype=np.float32).tobytes(), raw=True)
    add = helper.make_node("Add", ["x","c"], ["y"], name="")
    graph = helper.make_graph([add], "", [X], [Y])
    graph.initializer.extend([cval])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    m.producer_name=""; m.producer_version=""; m.domain=""
    m.model_version=0; m.doc_string=""
    return m

results["unfolded_constant_chain"] = sz(make_unfolded())
results["folded_constant_chain"] = sz(make_folded())

# ---- 9. Sharing initializers ----
def make_shared(opset=13, shared=True):
    X = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["n"])
    Y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["n"])
    w = np.array([1,2,3], dtype=np.float32)
    wv = helper.make_tensor("w", TensorProto.FLOAT, [3], w.tobytes(), raw=True)
    if shared:
        add1 = helper.make_node("Add", ["x","w"], ["t"], name="")
        add2 = helper.make_node("Add", ["t","w"], ["y"], name="")
        graph = helper.make_graph([add1, add2], "", [X], [Y])
        graph.initializer.extend([wv])
    else:
        wv2 = helper.make_tensor("w2", TensorProto.FLOAT, [3], w.tobytes(), raw=True)
        add1 = helper.make_node("Add", ["x","w"], ["t"], name="")
        add2 = helper.make_node("Add", ["t","w2"], ["y"], name="")
        graph = helper.make_graph([add1, add2], "", [X], [Y])
        graph.initializer.extend([wv, wv2])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    m.producer_name=""; m.producer_version=""; m.domain=""
    m.model_version=0; m.doc_string=""
    return m

results["shared_initializer"] = sz(make_shared(shared=True))
results["unshared_initializer"] = sz(make_shared(shared=False))

# ---- 10. Identity node elimination ----
def make_with_identity(opset=13):
    X = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["n"])
    Y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["n"])
    id1 = helper.make_node("Identity", ["x"], ["t1"], name="")
    id2 = helper.make_node("Identity", ["t1"], ["t2"], name="")
    id3 = helper.make_node("Identity", ["t2"], ["y"], name="")
    graph = helper.make_graph([id1,id2,id3], "", [X], [Y])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    m.producer_name=""; m.producer_version=""; m.domain=""
    m.model_version=0; m.doc_string=""
    return m

results["three_identity_nodes"] = sz(make_with_identity())
results["zero_identity_nodes"] = sz(make_identity(opset=13, name_doc=False))

# ---- 11. Empty graph ----
def make_empty_graph(opset=13):
    graph = helper.make_graph([], "", [], [])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    m.producer_name=""; m.producer_version=""; m.domain=""
    m.model_version=0; m.doc_string=""
    return m

m = make_empty_graph()
results["empty_graph_opset13"] = sz(m)

# ---- 12. metadata fields byte cost ----
m_full = make_identity(opset=13, name_doc=True)
m_min = make_identity(opset=13, name_doc=False)
results["identity_full_metadata"] = sz(m_full)
results["identity_stripped_metadata"] = sz(m_min)

print(json.dumps(results, indent=2, default=str))

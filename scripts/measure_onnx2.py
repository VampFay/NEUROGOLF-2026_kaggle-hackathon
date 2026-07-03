"""
Round 2: Deeper minimization tests.
- ClearField vs setting empty (proto3 presence semantics)
- Minimal graph name (1 char) since checker requires non-empty
- ONNX Runtime execution verification
- Protobuf raw byte editing (strip optional fields)
- Score formula simulation
"""
import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np
import os, json, io

OUT = "/home/z/my-project/data/onnx_research_out"
os.makedirs(OUT, exist_ok=True)

def sz(m):
    return len(m.SerializeToString())

def check(m):
    try:
        onnx.checker.check_model(m)
        return "ok"
    except Exception as e:
        return "FAIL: " + str(e)[:100]

def run(m, inputs):
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
        out = sess.run(None, inputs)
        return "runs", [getattr(o,'shape',None) for o in out]
    except Exception as e:
        return "FAIL", str(e)[:140]

results = {}

# ---- A. Graph name: empty vs 1-char vs default ----
def make_id(graph_name="g", clear_meta=False, opset=13):
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, ["n"])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["n"])
    node = helper.make_node("Identity", ["X"], ["Y"])
    graph = helper.make_graph([node], graph_name, [X], [Y])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    if clear_meta:
        m.ClearField("producer_name")
        m.ClearField("producer_version")
        m.ClearField("domain")
        m.ClearField("model_version")
        m.ClearField("doc_string")
        graph.ClearField("doc_string")
    return m

results["id_graphname_g"] = sz(make_id("g"))
results["id_graphname_1char"] = sz(make_id("a"))
results["id_graphname_8char"] = sz(make_id("graph123"))
results["id_graphname_g_clearmeta"] = sz(make_id("g", clear_meta=True))
results["id_graphname_g_clearmeta_check"] = check(make_id("g", clear_meta=True))

# What does default make_model produce? Inspect.
m = make_id("g")
print("Default model fields:")
print("  producer_name:", repr(m.producer_name))
print("  producer_version:", repr(m.producer_version))
print("  domain:", repr(m.domain))
print("  model_version:", m.model_version)
print("  doc_string:", repr(m.doc_string))
print("  ir_version:", m.ir_version)
print("  graph.name:", repr(m.graph.name))
print("  graph.doc_string:", repr(m.graph.doc_string))
m2 = make_id("g", clear_meta=True)
print("After ClearField:")
print("  producer_name:", repr(m2.producer_name))
print("  HasField producer_name:", m2.HasField("producer_name"))

# ---- B. Run verification for key small models ----
print("\n--- Run verification ---")
# Identity
m = make_id("g", clear_meta=True)
r = run(m, {"X": np.array([1.0,2.0,3.0], dtype=np.float32)})
results["run_identity"] = r
print("identity:", r)

# Conv 1x1
def make_conv1x1(clear_meta=True, opset=13, n_in=10, n_out=10, with_bias=True):
    X = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", n_in, "h", "w"])
    Y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", n_out, "h", "w"])
    w = np.random.randn(n_out, n_in, 1, 1).astype(np.float32)
    b = np.zeros(n_out, dtype=np.float32)
    wv = helper.make_tensor("w", TensorProto.FLOAT, list(w.shape), w.tobytes(), raw=True)
    bv = helper.make_tensor("b", TensorProto.FLOAT, [n_out], b.tobytes(), raw=True)
    inputs = ["x","w"] + (["b"] if with_bias else [])
    node = helper.make_node("Conv", inputs, ["y"])
    graph = helper.make_graph([node], "g", [X], [Y])
    graph.initializer.extend([wv] + ([bv] if with_bias else []))
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    if clear_meta:
        m.ClearField("producer_name"); m.ClearField("producer_version")
        m.ClearField("domain"); m.ClearField("model_version"); m.ClearField("doc_string")
    return m

m = make_conv1x1()
results["conv1x1_clearmeta_size"] = sz(m)
results["conv1x1_clearmeta_check"] = check(m)
xin = np.random.randn(1,10,4,4).astype(np.float32)
r = run(m, {"x": xin})
results["run_conv1x1"] = r
print("conv1x1 size:", results["conv1x1_clearmeta_size"], "check:", results["conv1x1_clearmeta_check"], "run:", r)

# QLinearConv run
def make_qconv1x1(clear_meta=True, opset=13, n_in=10, n_out=10, with_bias=True):
    X = helper.make_tensor_value_info("x", TensorProto.UINT8, ["N", n_in, "h", "w"])
    Y = helper.make_tensor_value_info("y", TensorProto.UINT8, ["N", n_out, "h", "w"])
    w = np.random.randint(-127, 127, size=(n_out, n_in, 1, 1), dtype=np.int8)
    b = np.zeros(n_out, dtype=np.int32)
    wv = helper.make_tensor("w", TensorProto.INT8, list(w.shape), w.tobytes(), raw=True)
    bv = helper.make_tensor("b", TensorProto.INT32, [n_out], b.tobytes(), raw=True)
    x_scale = helper.make_tensor("xs", TensorProto.FLOAT, [], np.array(0.01, dtype=np.float32).tobytes(), raw=True)
    x_zp = helper.make_tensor("xz", TensorProto.UINT8, [], np.array(0, dtype=np.uint8).tobytes(), raw=True)
    w_scale = helper.make_tensor("ws", TensorProto.FLOAT, [], np.array(0.01, dtype=np.float32).tobytes(), raw=True)
    w_zp = helper.make_tensor("wz", TensorProto.INT8, [], np.array(0, dtype=np.int8).tobytes(), raw=True)
    y_scale = helper.make_tensor("ys", TensorProto.FLOAT, [], np.array(0.01, dtype=np.float32).tobytes(), raw=True)
    y_zp = helper.make_tensor("yz", TensorProto.UINT8, [], np.array(0, dtype=np.uint8).tobytes(), raw=True)
    if with_bias:
        inputs = ["x","xs","xz","w","ws","wz","b","ys","yz"]
        inits = [wv, bv, x_scale, x_zp, w_scale, w_zp, y_scale, y_zp]
    else:
        inputs = ["x","xs","xz","w","ws","wz","ys","yz"]
        inits = [wv, x_scale, x_zp, w_scale, w_zp, y_scale, y_zp]
    node = helper.make_node("QLinearConv", inputs, ["y"])
    graph = helper.make_graph([node], "g", [X], [Y])
    graph.initializer.extend(inits)
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    if clear_meta:
        m.ClearField("producer_name"); m.ClearField("producer_version")
        m.ClearField("domain"); m.ClearField("model_version"); m.ClearField("doc_string")
    return m

m = make_qconv1x1()
results["qconv1x1_clearmeta_size"] = sz(m)
results["qconv1x1_clearmeta_check"] = check(m)
xin = np.random.randint(0, 255, size=(1,10,4,4), dtype=np.uint8)
r = run(m, {"x": xin})
results["run_qconv1x1"] = r
print("qconv1x1 size:", results["qconv1x1_clearmeta_size"], "check:", results["qconv1x1_clearmeta_check"], "run:", r)

m = make_qconv1x1(with_bias=False)
results["qconv1x1_nobias_clearmeta_size"] = sz(m)
r = run(m, {"x": xin})
results["run_qconv1x1_nobias"] = r
print("qconv1x1 nobias size:", results["qconv1x1_nobias_clearmeta_size"], "run:", r)

# ---- C. Score formula simulation ----
# Score = max(1, 25 - ln(params + bytes))
import math
def score(params, bytes_):
    return max(1.0, 25.0 - math.log(params + bytes_))

print("\n--- Score simulation ---")
# conv1x1: 100 params (weights) + 10 params (bias) = 110 params
print(f"conv1x1 f32 (110 params, 577 bytes): score = {score(110, 577):.3f}")
print(f"conv1x1 f32 nobias (100 params, 523 bytes): score = {score(100, 523):.3f}")
print(f"conv1x1 int8 (100 params, 327 bytes): score = {score(100, 327):.3f}")
print(f"conv3x3 f32 (910 params, 3777 bytes): score = {score(910, 3777):.3f}")
print(f"conv3x3 int8 (900 params, 1183 bytes): score = {score(900, 1183):.3f}")
print(f"identity (0 params, 67 bytes): score = {score(0, 67):.3f}")
print(f"empty (0 params, 22 bytes): score = {score(0, 22):.3f}")

# Theoretical: how small to hit score 20, 21, 22, 23, 24, 25?
print("\n--- Bytes budget for target scores (params=0) ---")
for target in [25, 24, 23, 22, 21, 20, 19, 18]:
    # 25 - target = ln(bytes) => bytes = e^(25-target)
    b = math.exp(25 - target)
    print(f"  score {target}: bytes <= {b:.1f}")

# ---- D. Protobuf raw inspection ----
print("\n--- Protobuf raw inspection ---")
m = make_id("g", clear_meta=True)
raw = m.SerializeToString()
print(f"identity clearmeta raw len: {len(raw)}")
print(f"hex (first 80): {raw[:80].hex()}")
# Decode wire format manually
def decode_varint(data, i):
    result = 0; shift = 0
    while True:
        b = data[i]; i += 1
        result |= (b & 0x7f) << shift
        if not (b & 0x80): break
        shift += 7
    return result, i

print("\nManual protobuf field decode of identity (clearmeta):")
i = 0
while i < len(raw):
    tag, i = decode_varint(raw, i)
    field_num = tag >> 3
    wire_type = tag & 7
    type_names = {0:'varint',1:'fixed64',2:'len',5:'fixed32'}
    tn = type_names.get(wire_type, f'wt{wire_type}')
    if wire_type == 0:
        val, i = decode_varint(raw, i)
        print(f"  field={field_num} {tn} val={val}")
    elif wire_type == 2:
        ln, i = decode_varint(raw, i)
        payload = raw[i:i+ln]; i += ln
        # try as string
        try:
            s = payload.decode('utf-8')
            if all(32 <= c < 127 or c in (10,13) for c in payload):
                print(f"  field={field_num} {tn} len={ln} str={s!r}")
            else:
                print(f"  field={field_num} {tn} len={ln} bytes={payload[:20].hex()}{'...' if ln>20 else ''}")
        except:
            print(f"  field={field_num} {tn} len={ln} bytes={payload[:20].hex()}{'...' if ln>20 else ''}")
    elif wire_type == 5:
        val = raw[i:i+4]; i += 4
        print(f"  field={field_num} {tn} bytes={val.hex()}")
    elif wire_type == 1:
        val = raw[i:i+8]; i += 8
        print(f"  field={field_num} {tn} bytes={val.hex()}")
    else:
        print(f"  field={field_num} UNKNOWN wt={wire_type}")
        break

print("\n" + json.dumps(results, indent=2, default=str))

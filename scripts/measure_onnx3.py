"""
Round 3: Realistic minimization with competition constraints.
- I/O names fixed: "input" / "output" (5 chars each)
- Must pass onnx.checker + run in ORT
- Statically-defined shapes (1,10,30,30)
- Compare team's DSL primitives vs minimized versions
- Test onnxsim
- Measure all ops the team uses
"""
import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np
import onnxruntime as ort
import onnxsim
import math, sys, json

sys.path.insert(0, '/home/z/my-project')
from neurogolf.dsl import (color_map, single_layer_conv2d, identity, argmax_over_channels,
                           mask_apply, count_params, model_size_bytes, model_cost, model_score)

IN = "input"; OUT = "output"
OPSET = 17
IR = 8

def score(p, b): return max(1.0, 25.0 - math.log(p + b))

def base_model(nodes, inits, ops=OPSET, ir=IR):
    X = helper.make_tensor_value_info(IN, TensorProto.FLOAT, [1,10,30,30])
    Y = helper.make_tensor_value_info(OUT, TensorProto.FLOAT, [1,10,30,30])
    g = helper.make_graph(nodes, "g", [X], [Y])
    g.initializer.extend(inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", ops)])
    m.ir_version = ir
    m.ClearField("producer_name"); m.ClearField("producer_version")
    m.ClearField("domain"); m.ClearField("doc_string")
    return m

def verify(m, xin=None):
    try: onnx.checker.check_model(m); ck="ok"
    except Exception as e: ck="CK:"+str(e)[:50]
    try:
        s = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
        if xin is None: xin = np.zeros((1,10,30,30),dtype=np.float32)
        s.run(None, {IN: xin}); r="runs"
    except Exception as e: r="RT:"+str(e)[:50]
    return ck, r

results = {}

# ============================================================
# A. color_map: team vs minimized (input/output names fixed)
# ============================================================
print("=== A. color_map (input/output names FIXED) ===")
m_team = color_map({0:1, 1:2})
ck, r = verify(m_team)
print(f"  team:        params={count_params(m_team)} bytes={model_size_bytes(m_team)} score={model_score(m_team):.3f} [{ck}|{r}]")

# minimized f32: no attrs, 1-char init name, stripped meta
def cm_min_f32():
    W = np.zeros((10,10,1,1),dtype=np.float32)
    mp={0:1,1:2}; full={c:mp.get(c,c) for c in range(10)}
    for f,t in full.items(): W[t,f,0,0]=1.0
    wv = numpy_helper.from_array(W, name="w")
    node = helper.make_node("Conv", [IN,"w"], [OUT])  # no attrs!
    return base_model([node], [wv])
m = cm_min_f32()
ck, r = verify(m)
print(f"  min f32:     params={count_params(m)} bytes={model_size_bytes(m)} score={model_score(m):.3f} [{ck}|{r}]")
results["color_map_team"] = model_size_bytes(m_team)
results["color_map_min_f32"] = model_size_bytes(m)

# int8 nobias QLinearConv (input is uint8 - but competition feeds float32!)
# Problem: competition feeds float32 (1,10,30,30). QLinearConv needs uint8/int8 input.
# So we'd need QuantizeLinear first. That adds overhead. Let me measure.
def cm_min_int8():
    W = np.zeros((10,10,1,1),dtype=np.int8)
    mp={0:1,1:2}; full={c:mp.get(c,c) for c in range(10)}
    for f,t in full.items(): W[t,f,0,0]=1
    wv = helper.make_tensor("w", TensorProto.INT8, [10,10,1,1], W.tobytes(), raw=True)
    # QuantizeLinear: float32 -> uint8
    def ms(nm,dt,v):
        a=np.array([v],dtype={TensorProto.FLOAT:np.float32,TensorProto.UINT8:np.uint8,TensorProto.INT8:np.int8}[dt])
        return helper.make_tensor(nm,dt,[1],a.tobytes(),raw=True)
    s1=ms("a",TensorProto.FLOAT,1.0/255); z1=ms("b",TensorProto.UINT8,0)
    s2=ms("c",TensorProto.FLOAT,1.0); z2=ms("d",TensorProto.INT8,0)
    s3=ms("e",TensorProto.FLOAT,1.0); z3=ms("f",TensorProto.UINT8,0)
    nodes = [
        helper.make_node("QuantizeLinear",[IN,"a","b"],["q"]),
        helper.make_node("QLinearConv",["q","a","b","w","c","d","e","f"],["r"]),
        helper.make_node("DequantizeLinear",["r","e","f"],[OUT]),
    ]
    return base_model(nodes, [wv,s1,z1,s2,z2,s3,z3], ops=13)  # QLinearConv needs opset>=10
m = cm_min_int8()
ck, r = verify(m)
print(f"  min int8:    params={count_params(m)} bytes={model_size_bytes(m)} score={model_score(m):.3f} [{ck}|{r}]")
results["color_map_min_int8"] = model_size_bytes(m)

# ============================================================
# B. single_layer_conv2d 3x3: team vs minimized
# ============================================================
print("\n=== B. single_layer_conv2d 3x3 (900 params) ===")
W = np.zeros((10,10,3,3),dtype=np.float32)
for i in range(10): W[i,i,1,1]=1.0
b = np.zeros(10,dtype=np.float32)
m_team = single_layer_conv2d(W, b)
ck, r = verify(m_team)
print(f"  team:        params={count_params(m_team)} bytes={model_size_bytes(m_team)} score={model_score(m_team):.3f} [{ck}|{r}]")

def conv3_min():
    wv = numpy_helper.from_array(W.astype(np.float32), name="w")
    bv = numpy_helper.from_array(b.astype(np.float32), name="b")
    node = helper.make_node("Conv", [IN,"w","b"], [OUT])  # no attrs! 3x3 -> 28x28 though
    return base_model([node], [wv,bv])
m = conv3_min()
ck, r = verify(m)
print(f"  min no-pad:  params={count_params(m)} bytes={model_size_bytes(m)} score={model_score(m):.3f} [{ck}|{r}] (out 28x28)")

def conv3_min_pad():
    wv = numpy_helper.from_array(W.astype(np.float32), name="w")
    bv = numpy_helper.from_array(b.astype(np.float32), name="b")
    node = helper.make_node("Conv", [IN,"w","b"], [OUT], kernel_shape=[3,3], pads=[1,1,1,1])
    return base_model([node], [wv,bv])
m = conv3_min_pad()
ck, r = verify(m)
print(f"  min w/pad:   params={count_params(m)} bytes={model_size_bytes(m)} score={model_score(m):.3f} [{ck}|{r}] (out 30x30)")
results["conv3x3_team"] = model_size_bytes(m_team)
results["conv3x3_min_pad"] = model_size_bytes(m)

# ============================================================
# C. identity: team vs minimized
# ============================================================
print("\n=== C. identity ===")
m_team = identity()
ck, r = verify(m_team)
print(f"  team:        params={count_params(m_team)} bytes={model_size_bytes(m_team)} score={model_score(m_team):.3f} [{ck}|{r}]")
def id_min():
    node = helper.make_node("Identity", [IN], [OUT])
    return base_model([node], [])
m = id_min()
ck, r = verify(m)
print(f"  min:         params={count_params(m)} bytes={model_size_bytes(m)} score={model_score(m):.3f} [{ck}|{r}]")
results["identity_team"] = model_size_bytes(m_team)
results["identity_min"] = model_size_bytes(m)

# ============================================================
# D. argmax_over_channels: team vs minimized
# ============================================================
print("\n=== D. argmax_over_channels ===")
m_team = argmax_over_channels()
ck, r = verify(m_team, xin=np.random.randn(1,10,30,30).astype(np.float32))
print(f"  team:        params={count_params(m_team)} bytes={model_size_bytes(m_team)} score={model_score(m_team):.3f} [{ck}|{r}]")
# Minimized: use initializers instead of Constant nodes, short names, drop keepdims default
def am_min():
    depth = helper.make_tensor("d", TensorProto.INT64, [], np.array(10,dtype=np.int64).tobytes(), raw=True)
    vals = helper.make_tensor("v", TensorProto.FLOAT, [2], np.array([0,1],dtype=np.float32).tobytes(), raw=True)
    nodes = [
        helper.make_node("ArgMax", [IN], ["a"], axis=1, keepdims=0),  # (1,30,30) int64
        helper.make_node("OneHot", ["a","d","v"], [OUT], axis=1),  # (1,10,30,30)
    ]
    return base_model(nodes, [depth, vals])
m = am_min()
ck, r = verify(m, xin=np.random.randn(1,10,30,30).astype(np.float32))
print(f"  min:         params={count_params(m)} bytes={model_size_bytes(m)} score={model_score(m):.3f} [{ck}|{r}]")
results["argmax_team"] = model_size_bytes(m_team)
results["argmax_min"] = model_size_bytes(m)

# ============================================================
# E. Individual op sizes (minimal, with input/output names)
# ============================================================
print("\n=== E. Individual op sizes (input/output fixed names) ===")
def op_size(nodes, inits, label, ops=OPSET):
    m = base_model(nodes, inits, ops=ops)
    ck, r = verify(m)
    b = model_size_bytes(m)
    p = count_params(m)
    print(f"  {label:30s}: params={p} bytes={b} score={score(p,b):.3f} [{ck}|{r}]")
    return b

# Transpose
op_size([helper.make_node("Transpose",[IN],[OUT],perm=[0,2,3,1])], [], "Transpose NCHW->NHWC")
op_size([helper.make_node("Transpose",[IN],[OUT],perm=[0,1,2,3])], [], "Transpose identity perm")

# Slice (opset 10 attr-based is smaller but we use 17; need input-based)
def slice_op():
    s=helper.make_tensor("s",TensorProto.INT64,[1],np.array([0],dtype=np.int64).tobytes(),raw=True)
    e=helper.make_tensor("e",TensorProto.INT64,[1],np.array([30],dtype=np.int64).tobytes(),raw=True)
    a=helper.make_tensor("a",TensorProto.INT64,[1],np.array([2],dtype=np.int64).tobytes(),raw=True)
    st=helper.make_node("Slice",[IN,"s","e","a"],[OUT])
    return [st],[s,e,a]
op_size(*slice_op(), "Slice [0:30] axis 2")

# Slice + Concat
def slice_concat():
    s=helper.make_tensor("s",TensorProto.INT64,[1],np.array([0],dtype=np.int64).tobytes(),raw=True)
    e=helper.make_tensor("e",TensorProto.INT64,[1],np.array([15],dtype=np.int64).tobytes(),raw=True)
    a=helper.make_tensor("a",TensorProto.INT64,[1],np.array([2],dtype=np.int64).tobytes(),raw=True)
    n1=helper.make_node("Slice",[IN,"s","e","a"],["p"])
    n2=helper.make_node("Concat",["p","p"],[OUT],axis=2)
    return [n1,n2],[s,e,a]
op_size(*slice_concat(), "Slice+Concat")

# Resize nearest
def resize_nn():
    sc=helper.make_tensor("s",TensorProto.FLOAT,[4],np.array([1,1,2,2],dtype=np.float32).tobytes(),raw=True)
    roi=helper.make_tensor("r",TensorProto.FLOAT,[0],b"",raw=True)
    n=helper.make_node("Resize",[IN,"r","s"],[OUT],mode="nearest")
    return [n],[roi,sc]
op_size(*resize_nn(), "Resize nn 2x (opset17)", ops=13)

# Pad
def pad_op():
    # Pad pads input is 1D [2*ndim] = [0,0,1,1,0,0,1,1] for 4D (begin_dims then end_dims)
    p=helper.make_tensor("p",TensorProto.INT64,[8],np.array([0,0,1,1,0,0,1,1],dtype=np.int64).tobytes(),raw=True)
    n=helper.make_node("Pad",[IN,"p"],[OUT])
    return [n],[p]
op_size(*pad_op(), "Pad 1px (opset13)", ops=13)

# Mul (with scalar const)
def mul_op():
    c=helper.make_tensor("c",TensorProto.FLOAT,[],np.array(2.0,dtype=np.float32).tobytes(),raw=True)
    n=helper.make_node("Mul",[IN,"c"],[OUT])
    return [n],[c]
op_size(*mul_op(), "Mul by scalar 2.0")

# Add (with scalar const)
def add_op():
    c=helper.make_tensor("c",TensorProto.FLOAT,[],np.array(1.0,dtype=np.float32).tobytes(),raw=True)
    n=helper.make_node("Add",[IN,"c"],[OUT])
    return [n],[c]
op_size(*add_op(), "Add scalar 1.0")

# Cast
op_size([helper.make_node("Cast",[IN],[OUT],to=TensorProto.FLOAT)], [], "Cast to float")

# Tile
def tile_op():
    r=helper.make_tensor("r",TensorProto.INT64,[4],np.array([1,1,1,1],dtype=np.int64).tobytes(),raw=True)
    n=helper.make_node("Tile",[IN,"r"],[OUT])
    return [n],[r]
op_size(*tile_op(), "Tile 1x (noop)")

# ============================================================
# F. onnxsim test
# ============================================================
print("\n=== F. onnxsim ===")
# Build a model with redundant Identity + Constant chain, then simplify
def bloated():
    cval = helper.make_tensor("c", TensorProto.FLOAT, [3], np.array([1,2,3],dtype=np.float32).tobytes(), raw=True)
    nodes = [
        helper.make_node("Constant", [], ["c"], value=cval),
        helper.make_node("Identity", ["c"], ["c2"]),
        helper.make_node("Add", [IN, "c2"], [OUT]),  # but IN is (1,10,30,30), c2 is (3,) - won't work
    ]
    # Fix: make c broadcastable
    return base_model(nodes[:2], [])  # just constant+identity
m_bloated = bloated()
print(f"  bloated (Const+Identity): {model_size_bytes(m_bloated)} bytes")
try:
    m_sim, ok = onnxsim.simplify(m_bloated)
    if ok:
        print(f"  onnxsim output:          {model_size_bytes(m_sim)} bytes")
    else:
        print(f"  onnxsim returned ok=False")
except Exception as e:
    print(f"  onnxsim failed: {e}")

# onnxsim on team's color_map
try:
    m_sim, ok = onnxsim.simplify(m_team)
    print(f"  team color_map: {model_size_bytes(m_team)} -> onnxsim: {model_size_bytes(m_sim)} bytes (ok={ok})")
except Exception as e:
    print(f"  onnxsim on color_map failed: {e}")

# ============================================================
# G. Protobuf field inventory of minimal identity
# ============================================================
print("\n=== G. Protobuf field inventory (minimal identity, 67 bytes) ===")
m = id_min()
raw = m.SerializeToString()
print(f"  total: {len(raw)} bytes")
# Decode
def dv(data,i):
    r=0;s=0
    while True:
        b=data[i];i+=1;r|=(b&0x7f)<<s
        if not(b&0x80):break
        s+=7
    return r,i
i=0; fields=[]
while i<len(raw):
    tag,i=dv(raw,i); fn=tag>>3; wt=tag&7
    if wt==0:
        v,i=dv(raw,i); fields.append((fn,wt,v))
    elif wt==2:
        ln,i=dv(raw,i); fields.append((fn,wt,ln)); i+=ln
    elif wt==5: fields.append((fn,wt,4)); i+=4
    elif wt==1: fields.append((fn,wt,8)); i+=8
# ModelProto fields: 1=ir_version,7=opset_import,8=graph
print("  Top-level ModelProto fields:")
for fn,wt,v in fields:
    print(f"    field={fn} wt={wt} {'val='+str(v) if wt==0 else 'len='+str(v)}")

print("\n" + json.dumps(results, indent=2))

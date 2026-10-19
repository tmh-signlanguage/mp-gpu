"""Export a .tflite graph to a simple JSON op list + npz of dense float32 constants.

Handles sparse (DENSIFY) and fp16 (DEQUANTIZE) weights by running the interpreter
and reading back their outputs. Needs TensorFlow (e.g. tensorflow-cpu==2.15). Only required to re-create src/mp_gpu/models/ from the
MediaPipe .tflite files:  python tools/tflite_export.py pose_detection.tflite
Produces <model>.graph.json and <model>.weights.npz (float32; the shipped files store fp16 where exact).
"""
import sys, json
import numpy as np
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as fb

path = sys.argv[1]
buf = open(path, "rb").read()
model = fb.ModelT.InitFromObj(fb.Model.GetRootAsModel(buf, 0))
sg = model.subgraphs[0]
opnames = {v: k for k, v in vars(fb.BuiltinOperator).items() if not k.startswith("_")}
act_names = {v: k for k, v in vars(fb.ActivationFunctionType).items() if not k.startswith("_")}
pad_names = {v: k for k, v in vars(fb.Padding).items() if not k.startswith("_")}

it = tf.lite.Interpreter(model_path=path, experimental_preserve_all_tensors=True,
                         experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
it.allocate_tensors()
inp = it.get_input_details()[0]
it.set_tensor(inp["index"], np.zeros(inp["shape"], np.float32))
it.invoke()

# Tensors produced by DENSIFY/DEQUANTIZE, plus buffer-backed float constants, become weights.
const_ops = {"DENSIFY", "DEQUANTIZE"}
weights, ops = {}, []
produced_by_const = set()
for op in sg.operators:
    name = opnames[model.operatorCodes[op.opcodeIndex].builtinCode
                   if model.operatorCodes[op.opcodeIndex].builtinCode else
                   model.operatorCodes[op.opcodeIndex].deprecatedBuiltinCode]
    if name in const_ops:
        o = int(op.outputs[0])
        weights[str(o)] = it.get_tensor(o).astype(np.float32)
        produced_by_const.add(o)
        continue
    opts = {}
    o = op.builtinOptions
    if o is not None:
        for k, v in vars(o).items():
            if isinstance(v, (int, float, bool)):
                opts[k] = v
            elif isinstance(v, np.ndarray):
                opts[k] = v.tolist()
        if "fusedActivationFunction" in opts:
            opts["fusedActivationFunction"] = act_names[opts["fusedActivationFunction"]]
        if "padding" in opts:
            opts["padding"] = pad_names[opts["padding"]]
    ops.append({"op": name, "inputs": [int(i) for i in op.inputs], "outputs": [int(i) for i in op.outputs],
                "opts": opts})

for i, t in enumerate(sg.tensors):
    b = model.buffers[t.buffer].data
    if b is not None and len(b) and i not in produced_by_const and str(i) not in weights:
        try:
            weights[str(i)] = it.get_tensor(i).astype(np.float32)
        except ValueError:
            pass

shapes = {str(i): [int(x) for x in t.shape] for i, t in enumerate(sg.tensors)}
names = {str(i): t.name.decode() for i, t in enumerate(sg.tensors)}
graph = {"inputs": [int(i) for i in sg.inputs], "outputs": [int(i) for i in sg.outputs],
         "ops": ops, "shapes": shapes, "names": names}
base = path[:-len(".tflite")]
json.dump(graph, open(base + ".graph.json", "w"))
np.savez(base + ".weights.npz", **weights)
# Reference output for a fixed random input, to validate the torch port.
x = np.random.RandomState(0).uniform(-1, 1, inp["shape"]).astype(np.float32)
it.set_tensor(inp["index"], x); it.invoke()
np.savez(base + ".testio.npz", x=x, **{f"y{k}": it.get_tensor(d["index"]) for k, d in enumerate(it.get_output_details())})
print("ops:", sorted({o["op"] for o in ops}), "weights:", len(weights))

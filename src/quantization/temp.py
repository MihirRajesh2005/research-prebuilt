import onnx
from collections import Counter

model = onnx.load(
    r"C:\Users\dragon\projects\research-prebuilt\src\models\fp16\rt_hdr_alb_nrm.onnx",
    load_external_data=False
)

print("Inputs:")
for x in model.graph.input:
    print(x.name, x.type.tensor_type.elem_type)

print("\nOutputs:")
for x in model.graph.output:
    print(x.name, x.type.tensor_type.elem_type)

print("\nInitializers:")
print(Counter(i.data_type for i in model.graph.initializer))
from pathlib import Path
import onnx
from onnx import TensorProto
from onnx.external_data_helper import convert_model_to_external_data

root_dir = Path(__file__).resolve().parents[2]
fp16_model = root_dir / "src" / "models" / "fp16" / "rt_hdr_alb_nrm.onnx"
fp32_model = root_dir / "src" / "models" / "fp32" / "rt_hdr_alb_nrm.onnx"

def convert(model: onnx.ModelProto) -> onnx.ModelProto:
    for io in (
        list(model.graph.input)
        + list(model.graph.output)
        + list(model.graph.value_info)
    ):
        type = io.type.tensor_type
        if type.elem_type == TensorProto.FLOAT16:
            type.elem_type = TensorProto.FLOAT


    for initializer in model.graph.initializer:
        if initializer.data_type == TensorProto.FLOAT16:
            array = onnx.numpy_helper.to_array(initializer)
            converted = array.astype("float32")

            new = onnx.numpy_helper.from_array(
                converted, name=initializer.name
            )
            initializer.CopyFrom(new)

    for node in model.graph.node:
        if node.op_type != "Cast":
            continue

        for attribute in node.attribute:
            if attribute.name == "to" and attribute.i == TensorProto.FLOAT16:
                attribute.i = TensorProto.FLOAT

    return model

model = onnx.load(fp16_model)
model = convert(model)

onnx.save_model(
    model, 
    fp32_model,
    save_as_external_data=True,
    all_tensors_to_one_file=True,
    location="rt_hdr_alb_nrm.onnx.data",
    size_threshold=0
)

onnx.checker.check_model(fp32_model)
print("fp32 model generated successfully")
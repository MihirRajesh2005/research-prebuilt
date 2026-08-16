from pathlib import Path
import numpy as np
import onnx
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static
)

root = Path(__file__).resolve().parents[2]
fp16_model = root / "src" / "models" / "fp16" / "rt_hdr_alb_nrm.onnx"
output_dir = root / "src" / "models" / "int8"
int8_model = output_dir / "rt_hdr_alb_nrm.onnx"
calib_dir = root / "data" / "calibration" / "samples"

class oidn_calibration_data_reader(CalibrationDataReader):
    def __init__(self):
        self.samples = sorted(calib_dir.glob("sample_*.npy"))

        if not self.samples:
            raise RuntimeError(f"No calibration samples found")

        self.index = 0

    def get_next(self):
        if self.index >= len(self.samples):
            return None
        sample_path  = self.samples[self.index]
        self.index += 1
        sample = np.load (sample_path)

        if sample.shape != (1,9,1088,1920):
            raise ValueError(f"{sample_path.name}: unexpected shape {sample.shape}")

        if sample.dtype != np.float16:
            sample = sample.astype(np.float16)

        if not np.isfinite(sample).all():
            raise ValueError(f'{sample_path.name} contains inf or nan')

        return {"input" : sample}

def main():
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration_files = sorted(calib_dir.glob("sample_*.npy"))
    for path in calibration_files:
        sample = np.load(path)
        data_reader = oidn_calibration_data_reader()

        quantize_static(
            model_input=str(fp16_model),
            model_output=str(int8_model),
            calibration_data_reader=data_reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
            calibrate_method=CalibrationMethod.Entropy,
            extra_options={"ActivationSymmetric" : True,
                           "WeightSymmetric" : True,
                           "QuantizeBias" : False},
        )

    model = onnx.load(int8_model, load_external_data=False)
    onnx.checker.check_model(model)

    q_nodes = sum(
        node.op_type == "QuantizeLinear"
        for node in model.graph.node
    )

    dq_nodes = sum(
        node.op_type == "DequantizeLinear"
        for node in model.graph.node
    )

    print(f"QuantizeLinear nodes:   {q_nodes}")
    print(f"DequantizeLinear nodes: {dq_nodes}")
    print(f"Model inputs:           {len(model.graph.input)}")
    print(f"Model outputs:          {len(model.graph.output)}")
    print("ONNX checker: PASSED")


if __name__ == "__main__":
    main()
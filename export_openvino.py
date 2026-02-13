import openvino as ov
import argparse
import os


def convert_onnx_to_openvino(onnx_path: str, output_xml_path: str, model_type: str):
    """
    model_type: "det" or "rec"
    """

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    print(f"Loading ONNX model from: {onnx_path}")
    core = ov.Core()
    ov_model = core.read_model(onnx_path)

    input_tensor = ov_model.input()
    input_name = input_tensor.get_any_name()

    print(f"Original input shape: {input_tensor.partial_shape}")

    # -------------------------
    # DET MODEL
    # -------------------------
    if model_type == "det":
        # (N, 3, H, W)
        ov_model.reshape(
            {
                input_name: ov.PartialShape(
                    [-1, 3, -1, -1]
                )
            }
        )

    # -------------------------
    # REC MODEL
    # -------------------------
    elif model_type == "rec":
        # (N, 3, 48, W)
        ov_model.reshape(
            {
                input_name: ov.PartialShape(
                    [-1, 3, 48, -1]
                )
            }
        )

    else:
        raise ValueError("model_type must be 'det' or 'rec'")

    print(f"Reshaped input shape: {ov_model.input().partial_shape}")

    # FP16 IR 생성
    ov.save_model(
        ov_model,
        output_xml_path,
        compress_to_fp16=True,
    )

    print(f"OpenVINO model saved to: {output_xml_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ONNX → OpenVINO Converter")

    parser.add_argument(
        "--onnx_path",
        type=str,
        required=True,
        help="Path to ONNX model file",
    )

    parser.add_argument(
        "--openvino_model_path",
        type=str,
        required=True,
        help="Output path (.xml)",
    )

    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["det", "rec"],
        help="Model type: det or rec",
    )

    args = parser.parse_args()

    convert_onnx_to_openvino(
        onnx_path=args.onnx_path,
        output_xml_path=args.openvino_model_path,
        model_type=args.model_type,
    )

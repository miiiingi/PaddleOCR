import argparse
import torch
import tensorrt as trt
from Calibrator import MyCalibrator


def build_trt_engine(
    onnx_file_path,
    engine_file_path,
    # dummy_data,
    max_workspace_size=1 << 28,
    fp16_mode=False,
    int8_mode=False,
    strip_weights=False,
):
    """
    ONNX 모델 파일을 TensorRT 엔진으로 변환하고 저장합니다.
    :param onnx_file_path: ONNX 모델 파일 경로
    :param engine_file_path: 저장할 TensorRT 엔진 파일 경로
    :param max_workspace_size: 엔진 빌드 시 사용할 최대 워크스페이스 크기 (예: 256MB)
    :param fp16_mode: FP16 모드 사용 여부 (사용 가능한 하드웨어인 경우 성능 향상)
    """
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED  # ★ 프로파일링 강화

    explicit_batch_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch_flag)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # ONNX 모델 파일 읽어오기 및 파싱
    with open(onnx_file_path, "rb") as model_file:
        onnx_model = model_file.read()
    if not parser.parse(onnx_model):
        print("ONNX 모델 파싱 중 오류가 발생했습니다:")
        for error_idx in range(parser.num_errors):
            print(parser.get_error(error_idx))
        raise RuntimeError("ONNX 모델을 파싱할 수 없습니다.")

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]

    for input in inputs:
        print(f"Input Model {input.name} shape: {input.shape} {input.dtype}")
    for output in outputs:
        print(f"Output Model {output.name} shape: {output.shape} {output.dtype}")

    input_name = inputs[0].name  # 보통 "input" 이라고 지정됨
    profile = builder.create_optimization_profile()

    if "det" in onnx_file_path:
        min_shape = (1, 3, 320, 320)
        opt_shape = (1, 3, 960, 960)
        max_shape = (1, 3, 4032, 4032)
    elif "rec" in onnx_file_path:
        min_shape = (1, 3, 48, 32)
        opt_shape = (32, 3, 48, 320)
        max_shape = (128, 3, 48, 320)

    print(
        f"Optimization Profile for '{input_name}': min={min_shape}, opt={opt_shape}, max={max_shape}"
    )
    profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    # 빌더 설정: workspace size 및 FP16 모드 활성화 (하드웨어 지원 시)
    # builder.max_workspace_size = max_workspace_size
    float16 = fp16_mode
    int8 = int8_mode
    if float16:
        print(f"fp16 activated!")
        config.set_flag(trt.BuilderFlag.FP16)
    elif int8:
        print(f"int8 activated!")
        config.set_flag(trt.BuilderFlag.INT8)

        # INT8 모드에서는 캘리브레이터를 반드시 등록해야 합니다.
        # calibration_data는 미리 준비한 numpy 배열 리스트 또는 배열을 전달합니다.
        input_shape = dummy_data.shape  # 첫 번째 입력 텐서의 shape 사용
        calibrator = MyCalibrator(dummy_data.cpu().numpy(), 1, input_shape)
        config.int8_calibrator = calibrator

    if strip_weights:
        config.set_flag(trt.BuilderFlag.STRIP_PLAN)

    # TensorRT 엔진 빌드 (엔진 빌드 시 모델 최적화 수행)
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError("TensorRT 엔진 빌드에 실패했습니다.")

    # 생성된 엔진을 파일로 직렬화 및 저장
    with open(engine_file_path, "wb") as f:
        f.write(engine_bytes)
    print(f"TensorRT 엔진이 성공적으로 저장되었습니다: {engine_file_path}")
    return engine_bytes


def main():
    parser = argparse.ArgumentParser(
        description="MLflow로 저장된 PyTorch 모델을 ONNX로 변환한 후 TensorRT 엔진으로 변환합니다."
    )
    parser.add_argument(
        "--onnx_file",
        type=str,
        default="output_onnx/det.onnx",
        help="저장할 ONNX 파일 경로",
    )
    parser.add_argument(
        "--engine_file",
        type=str,
        default="output_onnx/det.trt",
        help="저장할 TensorRT 엔진 파일 경로",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Weather To FP16",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Weather To INT8",
    )
    parser.add_argument(
        "--strip_weights",
        action="store_true",
        help="Weather To strip weights",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch Size",
    )

    args = parser.parse_args()

    # ONNX 모델을 TensorRT 엔진으로 변환 및 저장 (FP16 모드를 사용하도록 설정)
    build_trt_engine(
        args.onnx_file,
        args.engine_file,
        # dummy_input,
        max_workspace_size=1 << 28,
        fp16_mode=args.fp16,
        int8_mode=args.int8,
        strip_weights=args.strip_weights,
    )


if __name__ == "__main__":
    main()

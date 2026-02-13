import numpy as np
import tensorrt as trt
import pycuda.driver as cuda


class MyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(
        self, calibration_data, batch_size, input_shape, cache_file="calib.cache"
    ):
        # 캘리브레이션 데이터, 배치 사이즈, 입력 텐서 shape, 그리고 (옵션) 캐시 파일 경로를 초기화합니다.
        super(MyCalibrator, self).__init__()
        self.calibration_data = calibration_data  # numpy 배열 리스트 혹은 배열
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.cache_file = cache_file
        self.current_index = 0

        # 입력 데이터 크기에 맞는 GPU 메모리를 할당합니다.
        self.device_input = cuda.mem_alloc(
            trt.volume(input_shape) * self.batch_size * np.float32().itemsize
        )

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        # 모든 배치 데이터를 다 사용했다면 None을 리턴해서 캘리브레이션을 종료합니다.
        if self.current_index + self.batch_size > len(self.calibration_data):
            return None

        # 배치 데이터를 준비합니다.
        batch = self.calibration_data[
            self.current_index : self.current_index + self.batch_size
        ]
        batch = np.ascontiguousarray(batch, dtype=np.float32)

        # GPU 메모리에 데이터를 복사합니다.
        cuda.memcpy_htod(self.device_input, batch)
        self.current_index += self.batch_size
        return [int(self.device_input)]

    def read_calibration_cache(self):
        # 이전 캘리브레이션 캐시가 있다면 이를 로드하여 사용합니다.
        try:
            with open(self.cache_file, "rb") as f:
                return f.read()
        except IOError:
            return None

    def write_calibration_cache(self, cache):
        # 캘리브레이션 캐시를 저장합니다.
        with open(self.cache_file, "wb") as f:
            f.write(cache)
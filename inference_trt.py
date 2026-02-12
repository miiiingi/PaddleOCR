import os
import time
import re
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
from PIL import Image
from typing import List, Tuple, Dict
import torch
import torch.nn.functional as F
from db_postprocess import DBPostProcess

cuda.init()
device = cuda.Device(0)

ctx = device.retain_primary_context()
ctx.push()


class PaddleOCRTensorRT:
    """PaddleOCR TensorRT Inference Engine"""

    def __init__(
        self,
        det_engine_path: str,
        rec_engine_path: str,
        dict_path: str = None,
    ):
        """
        Args:
            det_engine_path: Detection TensorRT engine 파일 경로
            rec_engine_path: Recognition TensorRT engine 파일 경로
            dict_path: OCR 문자 사전 파일 경로
        """
        # TensorRT 로거
        self.logger = trt.Logger(trt.Logger.INFO)

        # Detection Engine 로드
        self.det_engine, self.det_context = self._load_engine(det_engine_path)

        # Recognition Engine 로드
        self.rec_engine, self.rec_context = self._load_engine(rec_engine_path)

        # 문자 사전 로드
        self.char_dict = self._load_dict(dict_path) if dict_path else None
        self.db_postprocess = DBPostProcess(
            thresh=0.3,
            box_thresh=0.6,
            max_candidates=1000,
            unclip_ratio=1.5,
            box_type="quad",
        )
        print(f"self det engine: {self.det_engine}")
        print(f"self rec engine: {self.rec_engine}")
        print(f"self det context: {self.det_context}")
        print(f"self rec context: {self.rec_context}")

        print(f"✅ Detection engine loaded: {det_engine_path}")
        print(f"✅ Recognition engine loaded: {rec_engine_path}")

    def _load_engine(
        self, engine_path: str
    ) -> Tuple[trt.ICudaEngine, trt.IExecutionContext]:
        """TensorRT 엔진 로드"""
        with open(engine_path, "rb") as f:
            engine_data = f.read()

        runtime = trt.Runtime(self.logger)
        engine = runtime.deserialize_cuda_engine(engine_data)
        context = engine.create_execution_context()

        return engine, context

    def _load_dict(self, dict_path: str) -> List[str]:
        """문자 사전 로드"""
        with open(dict_path, "r", encoding="utf-8") as f:
            chars = [line.strip() for line in f]
        return chars

    def _allocate_buffers(
        self,
        engine: trt.ICudaEngine,
        context: trt.IExecutionContext,
        N: int,
    ) -> Tuple[Dict, cuda.Stream, Dict]:
        """버퍼 할당 - 동적 shape 기반"""
        bindings = {}
        stream = cuda.Stream()
        output_tensors = {}

        for i in range(engine.num_io_tensors):
            binding = engine.get_tensor_name(i)

            # ⚠️ context에서 실제 설정된 shape 가져오기 (중요!)
            shape_engine = engine.get_tensor_shape(binding)
            shape = context.get_tensor_shape(binding)
            dtype = engine.get_tensor_dtype(binding)

            # -1이 있으면 에러
            if -1 in shape:
                raise RuntimeError(
                    f"Shape not fully specified for {binding}: {shape}. "
                    "Make sure to call set_input_shape() first!"
                )

            np_dtype = trt.nptype(dtype)

            print(f"📦 Binding: {binding}")
            print(f"   Shape: {shape}")
            print(f"   Shape Engine: {shape_engine}")
            print(f"   Dtype: {dtype}")

            if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                # Input 버퍼 - GPU 메모리만 할당
                size = int(np.prod(shape)) * np.dtype(np_dtype).itemsize
                device_mem = cuda.mem_alloc(size)
                bindings[binding] = device_mem
            else:
                # Output 버퍼 - Torch 텐서로 할당
                torch_dtype = torch.from_numpy(np.array([], dtype=np_dtype)).dtype
                output_tensor = torch.empty(
                    size=tuple(shape), dtype=torch_dtype, device="cuda"
                )
                output_tensors[binding] = output_tensor
                bindings[binding] = output_tensor.data_ptr()

        return bindings, stream, output_tensors

    def _do_inference(self, context: trt.IExecutionContext, stream: cuda.Stream):
        """추론 실행"""
        context.execute_async_v3(stream_handle=stream.handle)
        stream.synchronize()

    def _preprocess_det(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple]:
        """Detection 전처리 (좌표 어긋남 수정 버전)"""

        h, w = image.shape[:2]

        target_size = 960
        ratio = min(target_size / h, target_size / w)

        # 1️⃣ 실제 resize 크기
        resize_h = int(h * ratio)
        resize_w = int(w * ratio)

        resized = cv2.resize(image, (resize_w, resize_h))

        # 2️⃣ 32의 배수로 올림 (내림 ❌)
        pad_h = int(np.ceil(resize_h / 32) * 32)
        pad_w = int(np.ceil(resize_w / 32) * 32)

        # 3️⃣ padding (오른쪽, 아래쪽만)
        padded = np.zeros((pad_h, pad_w, 3), dtype=np.float32)
        padded[:resize_h, :resize_w, :] = resized.astype(np.float32)

        # 4️⃣ 정규화
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

        normalized = (padded / 255.0 - mean) / std

        # 5️⃣ CHW 변환
        img_tensor = normalized.transpose(2, 0, 1)
        img_tensor = np.expand_dims(img_tensor, axis=0).astype(np.float32)

        return img_tensor, ratio, (h, w)

    def _postprocess_det(
        self,
        pred: torch.Tensor,
        ratio: float,
        orig_size: Tuple[int, int],
        thresh: float = 0.3,
    ):
        """
        pred: (1,1,H,W) torch.cuda tensor
        """

        # 1️⃣ squeeze
        pred = pred[0, 0]  # (H,W) on CUDA

        # 2️⃣ threshold (GPU)
        binary = (pred > thresh).to(torch.uint8)

        # 3️⃣ optional: morphology (GPU)
        # 간단한 closing 효과
        kernel = torch.ones((1, 1, 3, 3), device=pred.device)
        binary = F.conv2d(
            binary.unsqueeze(0).unsqueeze(0).float(),
            kernel,
            padding=1,
        )
        binary = (binary > 0).to(torch.uint8)[0, 0]

        # 🔥 여기까지 GPU

        # 4️⃣ contour는 CPU에서 처리 (최소 데이터만 이동)
        binary_cpu = binary.cpu().numpy() * 255

        contours, _ = cv2.findContours(
            binary_cpu,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        boxes = []

        for contour in contours:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)

            box = box / ratio
            boxes.append(box.astype("int32"))

        return boxes

    def _preprocess_rec(
        self, image: np.ndarray, boxes: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Recognition 전처리"""
        cropped_images = []

        for box in boxes:
            # 박스 영역 크롭
            x_min = int(np.min(box[:, 0]))
            x_max = int(np.max(box[:, 0]))
            y_min = int(np.min(box[:, 1]))
            y_max = int(np.max(box[:, 1]))

            crop = image[y_min:y_max, x_min:x_max]

            # 리사이즈 (높이 48, 가로 비율 유지)
            h, w = crop.shape[:2]
            ratio = 48.0 / h
            new_w = int(w * ratio)

            resized = cv2.resize(crop, (new_w, 48))

            # 정규화
            mean = np.array([0.5, 0.5, 0.5]).reshape(1, 1, 3)
            std = np.array([0.5, 0.5, 0.5]).reshape(1, 1, 3)
            normalized = (resized.astype(np.float32) / 255.0 - mean) / std

            # 너비를 320으로 패딩
            target_w = 320
            if new_w < target_w:
                padded = np.zeros((48, target_w, 3), dtype=np.float32)
                padded[:, :new_w, :] = normalized
                normalized = padded
            else:
                normalized = normalized[:, :target_w, :]

            # CHW 변환
            img_tensor = normalized.transpose(2, 0, 1)
            cropped_images.append(img_tensor)

        return cropped_images

    def _postprocess_rec(
        self, preds: np.ndarray, boxes: List[np.ndarray]
    ) -> List[Tuple[str, float, np.ndarray]]:
        """Recognition 후처리"""
        results = []

        for i, box in enumerate(boxes):
            pred = preds[i]  # (T, C)

            # CTC 디코딩
            indices = np.argmax(pred, axis=-1)

            # 중복 제거 및 blank 제거
            text = []
            prev_idx = -1
            for idx in indices:
                if idx != prev_idx and idx != 0:  # 0은 blank
                    if self.char_dict and idx < len(self.char_dict):
                        text.append(self.char_dict[idx])
                prev_idx = idx

            text_str = "".join(text)

            # 신뢰도 계산
            max_probs = np.max(pred, axis=-1)
            confidence = float(np.mean(max_probs))

            results.append((text_str, confidence, box))

        return results

    def detect_text(self, image: np.ndarray):

        img_tensor, ratio, orig_size = self._preprocess_det(image)
        img_torch = torch.from_numpy(img_tensor).cuda().contiguous()

        input_name = None
        for i in range(self.det_engine.num_io_tensors):
            name = self.det_engine.get_tensor_name(i)
            if self.det_engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                input_name = name
                break

        N, C, H, W = img_tensor.shape
        self.det_context.set_input_shape(input_name, (N, C, H, W))

        bindings, stream, outputs = self._allocate_buffers(
            self.det_engine, self.det_context, N
        )

        for i in range(self.det_engine.num_io_tensors):
            binding = self.det_engine.get_tensor_name(i)
            self.det_context.set_tensor_address(binding, int(bindings[binding]))

        cuda.memcpy_dtod_async(
            dest=int(bindings[input_name]),
            src=int(img_torch.data_ptr()),
            size=img_torch.nbytes,
            stream=stream,
        )

        self._do_inference(self.det_context, stream)

        output_name = list(outputs.keys())[0]
        pred = outputs[output_name]  # torch cuda tensor (1,1,H,W)

        # 🔥 Paddle DBPostProcess 적용
        pred_cpu = pred.detach().cpu().numpy()

        src_h, src_w = orig_size
        shape_list = [[src_h, src_w, ratio, ratio]]

        outs_dict = {"maps": pred_cpu}

        post_result = self.db_postprocess(outs_dict, shape_list)

        boxes = post_result[0]["points"]

        return boxes

    def recognize_text(
        self, image: np.ndarray, boxes: List[np.ndarray]
    ) -> List[Tuple[str, float, np.ndarray]]:
        """텍스트 인식"""
        if len(boxes) == 0:
            return []

        # 전처리
        cropped_images = self._preprocess_rec(image, boxes)

        # 배치 처리
        batch_size = len(cropped_images)
        img_batch = np.array(cropped_images, dtype=np.float32)

        # 입력 shape 설정
        input_name = None
        for i in range(self.rec_engine.num_io_tensors):
            name = self.rec_engine.get_tensor_name(i)
            if self.rec_engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                input_name = name
                break

        N, C, H, W = img_batch.shape
        self.rec_context.set_input_shape(input_name, (N, C, H, W))

        # 버퍼 할당
        bindings, stream, outputs = self._allocate_buffers(
            self.rec_engine, self.rec_context, N
        )

        # 입력 데이터 복사
        for binding in self.rec_engine:
            self.rec_context.set_tensor_address(binding, int(bindings[binding]))

        img_torch = torch.from_numpy(img_batch).cuda()
        cuda.memcpy_dtod_async(
            dest=int(bindings[input_name]),
            src=int(img_torch.data_ptr()),
            size=img_torch.nbytes,
            stream=stream,
        )

        # 추론
        self._do_inference(self.rec_context, stream)

        # 출력 가져오기
        output_name = list(outputs.keys())[0]
        preds = outputs[output_name].cpu().numpy()

        # 후처리
        results = self._postprocess_rec(preds, boxes)

        return results

    def predict(self, image_path: str) -> List[Tuple[str, float, np.ndarray]]:
        """전체 OCR 파이프라인"""
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # 1. 텍스트 검출
        boxes = self.detect_text(image)

        # 2. 텍스트 인식
        results = self.recognize_text(image, boxes)

        return results


def main():
    """사용 예제"""
    # TensorRT 엔진 파일 경로
    det_engine_path = r"output_onnx/det.trt"
    rec_engine_path = r"output_onnx/rec.trt"
    dict_path = r"ppocr/utils/dict/ppocrv5_dict.txt"  # 옵션

    # OCR 엔진 초기화
    ocr = PaddleOCRTensorRT(
        det_engine_path=det_engine_path,
        rec_engine_path=rec_engine_path,
        dict_path=dict_path,
    )

    # 이미지 폴더 처리
    folder = r"/mnt/d/workspace/HENKEL/syringe_temp"

    # 정규식 패턴
    pattern = re.compile(
        r"(batch|batch number|number:|idh#|net weight|weight|no\.|P/N:|BATCH|"
        r"Syringe#:|EXP:|Net weight:|weight:|Storage Temp:|Temp:|IDH|#:|LOT#|"
        r"D.O.M:|SID#|Part No:|No:|Vender ID:|ID:|Lot No:|No:|Temp:|"
        r"Storage Temp:|DOM|NET WT:|WT:|P.O#|Hen)",
        re.IGNORECASE,
    )

    for filename in os.listdir(folder):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        image_path = os.path.join(folder, filename)
        print(f"\n{'='*60}")
        print(f"Processing: {image_path}")
        print(f"{'='*60}")

        # 추론
        start = time.time()
        results = ocr.predict(image_path)
        end = time.time()

        # 결과 출력
        for text, score, box in results:
            text_norm = text.lower().strip()
            if pattern.search(text):
                print(f"✅ text: {text}")
                print(f"   score: {score:.3f}")
                print(f"   box: {box.tolist()}")

        print(f"\n⏱️  Duration: {end - start:.3f}s")
        print(f"📊 Total detections: {len(results)}")


if __name__ == "__main__":
    try:
        main()
    finally:
        ctx.pop()

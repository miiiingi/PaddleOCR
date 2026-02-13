import cv2
import numpy as np
import openvino as ov
from typing import List, Tuple
import argparse
import os
from db_postprocess import DBPostProcess


class OpenVINOOCR:
    def __init__(
        self,
        det_model_path: str,
        rec_model_path: str,
        device: str = "AUTO",  # "CPU", "GPU", "AUTO"
        visualize: bool = False,
    ):
        self.visualize = visualize
        self.filename = ""
        self.core = ov.Core()

        # ------------------------
        # Detection Model
        # ------------------------
        self.det_model = self.core.read_model(det_model_path)
        self.det_compiled = self.core.compile_model(self.det_model, device)
        self.det_input_layer = self.det_compiled.input(0)
        self.det_output_layer = self.det_compiled.output(0)

        # ------------------------
        # Recognition Model
        # ------------------------
        self.rec_model = self.core.read_model(rec_model_path)
        self.rec_compiled = self.core.compile_model(self.rec_model, device)
        self.rec_input_layer = self.rec_compiled.input(0)
        self.rec_output_layer = self.rec_compiled.output(0)

        self.db_postprocess = DBPostProcess(
            thresh=0.3,
            box_thresh=0.6,
            max_candidates=1000,
            unclip_ratio=1.5,
            box_type="quad",
        )

        print("✅ OpenVINO models loaded successfully")

    def load_character_dict(self, dict_path):
        with open(dict_path, "r", encoding="utf-8") as f:
            chars = [line.strip() for line in f.readlines()]
        chars = [""] + chars  # blank for CTC
        return chars

    # =====================================================
    # DETECTION
    # =====================================================
    def preprocess_det(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple]:
        """Detection 전처리 (좌표 어긋남 수정 버전)"""
        h, w = image.shape[:2]

        target_size = 960

        ratio_h, ratio_w = target_size / h, target_size / w
        # 1️⃣ 실제 resize 크기
        resize_h = int(h * ratio_h)
        resize_w = int(w * ratio_w)
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
        return img_tensor, ratio_h, ratio_w, (h, w)

    def postprocess_det(self, det_map, thresh: float = 0.3):
        binary = (det_map > thresh).astype(np.uint8) * 255

        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append(np.array([x, y, x + w, y + h]))

        return boxes

    def detect(self, image: np.ndarray) -> np.ndarray:
        input_tensor, ratio_h, ratio_w, original_size = self.preprocess_det(image)

        result = self.det_compiled([input_tensor])
        output = result[self.det_output_layer]  # output shape: (1,1,H,W)
        src_h, src_w = original_size
        shape_list = [[src_h, src_w, ratio_h, ratio_w]]

        outs_dict = {"maps": output}

        post_result = self.db_postprocess(outs_dict, shape_list)

        boxes = post_result[0]["points"]

        if self.visualize:
            self.draw_detect_results(image, boxes)

        return boxes

    # =====================================================
    # RECOGNITION
    # =====================================================
    def get_rotate_crop_image(self, img, points):
        """
        4개의 점(x, y)으로 된 박스를 받아 수평으로 펴서(Warp) 잘라내는 함수
        points shape: (4, 2)
        """
        # 1. 점의 순서를 정렬 (좌상, 우상, 우하, 좌하 순서로 보정)
        # x좌표 기준으로 정렬
        sorted_p = points[np.argsort(points[:, 0])]

        # 가장 왼쪽 2개 점 중 y가 작은게 좌상(TL), 큰게 좌하(BL)
        left_half = sorted_p[:2]
        left_half = left_half[np.argsort(left_half[:, 1])]
        tl, bl = left_half

        # 가장 오른쪽 2개 점 중 y가 작은게 우상(TR), 큰게 우하(BR)
        right_half = sorted_p[2:]
        right_half = right_half[np.argsort(right_half[:, 1])]
        tr, br = right_half

        # 정렬된 좌표
        img_crop_width = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
        img_crop_height = int(max(np.linalg.norm(tl - bl), np.linalg.norm(tr - br)))

        pts_std = np.float32(
            [
                [0, 0],
                [img_crop_width, 0],
                [img_crop_width, img_crop_height],
                [0, img_crop_height],
            ]
        )

        # 소스 좌표 (float32로 변환 필수)
        src_pts = np.float32([tl, tr, br, bl])

        # 투시 변환 행렬 계산 및 적용
        M = cv2.getPerspectiveTransform(src_pts, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M,
            (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC,
        )

        # 세로로 긴 이미지(세로쓰기 등)일 경우 회전 (PP-OCR 로직상 가로로 눕혀야 함)
        if dst_img.shape[0] / dst_img.shape[1] > 1.5:
            dst_img = np.rot90(dst_img, k=1)

        return dst_img

    def preprocess_rec(self, image: np.ndarray, boxes: np.ndarray) -> List[np.ndarray]:
        """
        PP-OCRv5 Rec 모델 전처리 (Numpy array boxes 대응 버전)
        boxes shape: (N, 4, 2)
        """
        img_h = 48
        img_w = 320

        norm_img_batch = []

        # boxes가 리스트가 아닌 numpy array이므로 바로 순회
        for box in boxes:
            # box shape: (4, 2) -> [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

            # 1. Perspective Transform으로 텍스트 영역을 똑바로 펴서 자름
            crop = self.get_rotate_crop_image(image, box)

            if crop.size == 0:
                crop = np.zeros((img_h, img_w, 3), dtype=np.uint8)

            # 2. 비율 유지 리사이즈 (높이 48 고정)
            h, w = crop.shape[:2]
            ratio = w / float(h)

            new_w = int(img_h * ratio)
            if new_w > img_w:
                new_w = img_w

            resized = cv2.resize(crop, (new_w, img_h))

            # 3. 정규화: (Pixel - 0.5) / 0.5 => [-1, 1]
            resized = resized.astype(np.float32) / 255.0
            resized = (resized - 0.5) / 0.5

            # 4. Padding (오른쪽 채우기)
            padded = np.zeros((3, img_h, img_w), dtype=np.float32)
            resized = resized.transpose(2, 0, 1)  # HWC -> CHW
            padded[:, :, :new_w] = resized

            norm_img_batch.append(padded)

        return norm_img_batch

    def recognize(self, image: np.ndarray, boxes: List[np.ndarray]):

        crops = self.preprocess_rec(image, boxes)
        if len(crops) == 0:
            return []

        # 🔥 1. 가장 큰 width 찾기
        max_w = max(crop.shape[2] for crop in crops)

        padded_crops = []

        for crop in crops:
            c, h, w = crop.shape

            if w < max_w:
                pad_width = max_w - w
                pad = np.zeros((c, h, pad_width), dtype=np.float32)
                crop = np.concatenate([crop, pad], axis=2)

            padded_crops.append(crop)

        # 🔥 2. 이제 안전하게 batch 생성
        batch = np.stack(padded_crops, axis=0)

        result = self.rec_compiled([batch])
        output = result[self.rec_output_layer]

        return output

    def ctc_decode(self, preds, character_dict):
        """
        preds: (N, T, C) - (9, 40, 18385)
        character_dict: list (로드된 사전)
        """
        texts = []
        confs = []

        # 예측값에서 가장 높은 확률의 인덱스와 그 확률값을 가져옴
        preds_idx = preds.argmax(axis=2)  # (N, T)
        preds_prob = preds.max(axis=2)  # (N, T)

        for pred_idx, pred_prob in zip(preds_idx, preds_prob):
            char_list = []
            conf_list = []
            prev_idx = 0  # CTC의 blank index는 보통 0입니다.

            for i in range(len(pred_idx)):
                idx = int(pred_idx[i])
                prob = pred_prob[i]

                # 1. Blank(0)가 아니고, 이전 인덱스와 중복되지 않을 때만 추가
                if idx > 0 and (i == 0 or idx != pred_idx[i - 1]):
                    # character_dict 범위를 벗어나는지 체크 (안전장치)
                    if idx < len(character_dict):
                        char_list.append(character_dict[idx])
                        conf_list.append(prob)

            text = "".join(char_list)
            conf = np.mean(conf_list) if conf_list else 0.0

            texts.append(text)
            confs.append(float(conf))

        return texts, confs

    def draw_ocr_results(self, image, boxes, texts, confs):
        vis_img = image.copy()

        for box, text, conf in zip(boxes, texts, confs):
            # (N, 4, 2)에서 한 박스(4, 2)를 가져와 정수형 변환
            pts = box.astype(np.int32)

            # 1. 다각형(Polylines) 그리기
            # pts는 (4, 1, 2) 형태여야 cv2.polylines에서 인식함
            cv2.polylines(vis_img, [pts.reshape((-1, 1, 2))], True, (0, 255, 0), 2)

            # 2. 텍스트 위치 (박스 점들 중 가장 위쪽/왼쪽 기준)
            min_x = int(np.min(box[:, 0]))
            min_y = int(np.min(box[:, 1]))

            label = f"{text} ({conf:.2f})"
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )

            # 3. 텍스트 배경 사각형
            cv2.rectangle(
                vis_img, (min_x, min_y - th - 5), (min_x + tw, min_y), (0, 255, 0), -1
            )

            # 4. 텍스트 쓰기
            cv2.putText(
                vis_img,
                label,
                (min_x, min_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
        save_path = "detect_rec_result_" + self.filename

        cv2.imwrite(save_path, vis_img)
        print(f"✅ debug(detect_rec) image saved: {save_path}")

    def draw_detect_results(self, image, boxes):
        debug_img = image.copy()

        for i, box in enumerate(boxes):

            box = box.astype(np.int32)

            # 다각형 그리기
            cv2.polylines(
                debug_img,
                [box],
                isClosed=True,
                color=(0, 255, 0),
                thickness=2,
            )

            # 인덱스 표시
            cv2.putText(
                debug_img,
                str(i),
                tuple(box[0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
        save_path = "detect_result_" + self.filename
        cv2.imwrite(save_path, debug_img)
        print(f"✅ debug(detect) image saved: {save_path}")

    # =====================================================
    # FULL PIPELINE
    # =====================================================
    def predict(self, image_path: str):
        image = cv2.imread(image_path)
        character_dict = self.load_character_dict(r"ppocr/utils/dict/ppocrv5_dict.txt")

        # 1️⃣ Detect
        boxes = self.detect(image)

        # 2️⃣ Recognize
        rec_output = self.recognize(image, boxes)
        print(f"rec_output shape: {rec_output.shape}")

        text, confs = self.ctc_decode(rec_output, character_dict)

        self.draw_ocr_results(image, boxes, text, confs)

        return boxes, rec_output


# =====================================================
# 실행 예시
# =====================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ONNX → OpenVINO Converter")

    parser.add_argument(
        "--det_model_path",
        type=str,
        required=True,
        help="Input Detection model path (.xml)",
    )
    parser.add_argument(
        "--rec_model_path",
        type=str,
        required=True,
        help="Input Recognition model path (.xml)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize detection results",
    )

    args = parser.parse_args()

    ocr = OpenVINOOCR(
        args.det_model_path, args.rec_model_path, device="CPU", visualize=args.visualize
    )
    folder = r"/mnt/d/workspace/HENKEL/syringe_temp"
    for filename in os.listdir(folder):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        ocr.filename = filename
        image_path = os.path.join(folder, filename)
        print(f"\n{'='*60}")
        print(f"Processing: {image_path}")
        print(f"{'='*60}")

        boxes, rec_out = ocr.predict(image_path)

        print("Detected boxes:", len(boxes))
        print("Recognition output shape:", rec_out.shape if len(boxes) > 0 else None)

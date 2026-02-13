from paddleocr import PaddleOCR
import time
import re
import os
from tqdm import tqdm

# PaddleOCR 인스턴스 초기화
ocr = PaddleOCR(
    ocr_version="PP-OCRv5",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    # enable_hpi=True,
    # use_tensorrt=True,
    device="gpu:0",
)

# 샘플 이미지에 대해 OCR 추론 실행
folder = r"/mnt/d/workspace/HENKEL/syringe_temp"
for path in os.listdir(folder):
    folder_path = os.path.join(folder, path)
    print(f"image name: {folder_path}")
    result = ocr.predict(input=folder_path)
    pattern = re.compile(
        r"(batch|batch number|number:|idh#|net weight|weight|no\.|P/N:|BATCH|Syringe#:|EXP:|Net weight:|weight:|Storage Temp:|Temp:|IDH|#:|LOT#|D.O.M:|SID#|Part No:|No:|Vender ID:|ID:|Lot No:|No:|Temp:|Storage Temp:|DOM|NET WT:|WT:|P.O#|Hen)",
        re.IGNORECASE,
        # r"(Hen)", re.IGNORECASE
    )
    for res in result:
        res_value = res._to_json()["res"]
        res.save_to_img("output")

        for text, score, box in zip(
            res_value["rec_texts"], res_value["rec_scores"], res_value["rec_boxes"]
        ):
            text_norm = text.lower().strip()

            if not pattern.search(text):
                continue

            print(f"text: {text}, box: {box}, score: {score}")

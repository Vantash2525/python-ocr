import cv2
import os
import re
from paddleocr import PaddleOCR

def do_ocr(image_path):

    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

    # ---------------- OCR ----------------
    ocr = PaddleOCR(
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="en_PP-OCRv5_mobile_rec",
        det_db_thresh=0.5,
        det_db_box_thresh=0.5,
        use_textline_orientation=False
    )

    img = cv2.imread(image_path)
    if img is None:
        print("❌ Image not loaded")
        return []

    result = ocr.ocr(img)

    # ---------------- COLLECT TEXT + BOXES ----------------
    all_results = []  # stores { text, score, box } for every detected word
    buff = ""

    for line in result[0]:
        box = line[0]
        text, score = line[1]

        x      = int(box[0][0])
        y      = int(box[0][1])
        width  = int(box[2][0] - box[0][0])
        height = int(box[2][1] - box[0][1])

        all_results.append({
            "text": text,
            "score": round(score, 2),
            "box": { "x": x, "y": y, "width": width, "height": height }
        })

        print(f"📄 Detected: '{text}' (score: {score:.2f})")
        buff += text

    print("\n🔍 RAW TEXT:\n", buff)

    # ---------------- NORMALIZE ----------------
    buff = buff.replace("(", "{").replace(")", "}")
    buff = buff.replace("O", "0")
    buff = buff.replace("l", "1")
    buff = buff.replace("I", "1")

    print("\n🧵 MERGED TEXT:\n", buff)

    # ---------------- EXTRACT MATCHES ----------------
    matches = re.findall(r'ENC\{[a-f0-9]+\}', buff)

    if not matches:
        print("\n⚠️ No ENC{} pattern found, returning all OCR results")
        return all_results  # ✅ still return everything instead of exit()

    # ---------------- FILTER: only return boxes that matched ----------------
    print(f"\n✅ Found {len(matches)} matches:\n")

    output = []
    for item in all_results:
        # Normalize this item's text the same way
        normalized = item["text"].replace("(", "{").replace(")", "}") \
                                 .replace("O", "0").replace("l", "1").replace("I", "1")

        # Check if this word is part of any match
        for match in matches:
            if normalized in match or match in normalized:
                print(f"  ✅ {item['text']} → {item['box']}")
                output.append(item)  # ✅ keeps the full { text, score, box } dict
                break

    
    return output  # ✅ returns list of { text, score, box }
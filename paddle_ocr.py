import os
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_DISABLE_ONEDNN"] = "1"

import cv2
import re
import logging
from paddleocr import PaddleOCR

logging.getLogger("ppocr").setLevel(logging.ERROR)


def normalize_ocr_text(text):
    return (
        text.replace("(", "{")
            .replace(")", "}")
            .replace("O", "0")
            .replace("l", "1")
            .replace("I", "1")
            .replace(" ", "")
    )


def merge_boxes(items):
    min_x = min(item["box"]["x"] for item in items)
    min_y = min(item["box"]["y"] for item in items)
    max_x = max(item["box"]["x"] + item["box"]["width"] for item in items)
    max_y = max(item["box"]["y"] + item["box"]["height"] for item in items)

    return {
        "x": min_x,
        "y": min_y,
        "width": max_x - min_x,
        "height": max_y - min_y,
    }

def do_paddle_ocr(image_path):
    reader = PaddleOCR(
        ocr_version="PP-OCRv4",
        use_angle_cls=True,
        lang='en',
        use_gpu=False,
        show_log=False
    )

    img = cv2.imread(image_path)
    if img is None:
        print("❌ Image not loaded")
        return []

    result = reader.ocr(img, cls=True)

    # Handle None or empty result
    if not result or result[0] is None:
        print("⚠️ No text detected in image")
        return []

    page = result[0]

    # ---------------- COLLECT TEXT + BOXES ----------------
    all_results = []
    buff = ""

    for line in page:
        box, (text, score) = line

        x      = int(box[0][0])
        y      = int(box[0][1])
        width  = int(box[1][0] - box[0][0])
        height = int(box[2][1] - box[0][1])

        all_results.append({
            "text": text,
            "score": round(float(score), 2),
            "box": {"x": x, "y": y, "width": width, "height": height}
        })

        print(f"📄 Detected: '{text}' (score: {score:.2f})")
        buff += text

    print("\n🔍 RAW TEXT:\n", buff)

    # ---------------- NORMALIZE ----------------
    buff = normalize_ocr_text(buff)

    print("\n🧵 MERGED TEXT:\n", buff)

    # ---------------- EXTRACT MATCHES ----------------
    matches = re.findall(r'ENC\{[a-f0-9]+\}', buff)

    if not matches:
        print("\n⚠️ No ENC{} pattern found, returning all OCR results")
        return all_results

    # ---------------- FILTER ----------------
    print(f"\n✅ Found {len(matches)} matches:\n")

    output = []
    active_group = []
    active_text = ""

    for item in all_results:
        normalized = normalize_ocr_text(item["text"])

        if not active_group:
            if "ENC{" not in normalized:
                continue

            active_group = [item]
            active_text = normalized
        else:
            active_group.append(item)
            active_text += normalized

        match = re.search(r'ENC\{[a-f0-9]+\}', active_text)
        if match:
            merged_item = {
                "text": match.group(0),
                "score": round(min(entry["score"] for entry in active_group), 2),
                "box": merge_boxes(active_group),
            }
            print(f"  ✅ {merged_item['text']} → {merged_item['box']}")
            output.append(merged_item)
            active_group = []
            active_text = ""

    return output if output else all_results
import easyocr
import cv2
import re

def do_easy_ocr(image_path):
    reader = easyocr.Reader(['en'], gpu=False)

    img = cv2.imread(image_path)
    if img is None:
        print("❌ Image not loaded")
        return []

    result = reader.readtext(img)

    if not result:
        print("⚠️ No text detected in image")
        return []

    # ---------------- COLLECT TEXT + BOXES ----------------
    all_results = []
    buff = ""

    for (box, text, score) in result:
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
    buff = buff.replace("(", "{").replace(")", "}")
    buff = buff.replace("O", "0")
    buff = buff.replace("l", "1")
    buff = buff.replace("I", "1")

    print("\n🧵 MERGED TEXT:\n", buff)

    # ---------------- EXTRACT MATCHES ----------------
    matches = re.findall(r'ENC\{[a-f0-9]+\}', buff)

    if not matches:
        print("\n⚠️ No ENC{} pattern found, returning all OCR results")
        return all_results

    # ---------------- FILTER ----------------
    print(f"\n✅ Found {len(matches)} matches:\n")

    output = []
    for item in all_results:
        normalized = item["text"].replace("(", "{").replace(")", "}") \
                                 .replace("O", "0").replace("l", "1").replace("I", "1")
        for match in matches:
            if normalized in match or match in normalized:
                print(f"  ✅ {item['text']} → {item['box']}")
                output.append(item)
                break

    return output
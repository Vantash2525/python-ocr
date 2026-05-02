import os
from fastapi import FastAPI, File, UploadFile
from easyocr_model import do_easy_ocr
from paddle_ocr import do_paddle_ocr
import tempfile


app = FastAPI()

@app.post("/easy_ocr")
async def upload_file(file: UploadFile = File(...)):
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Call OCR function with file path
        result = do_easy_ocr(tmp_path)
        return {"filename": file.filename, "content_type": file.content_type, "ocr_results": result}
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.post("/paddle_ocr")
async def upload_file_paddle(file: UploadFile = File(...)):
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Call OCR function with file path
        result = do_paddle_ocr(tmp_path)
        return {"filename": file.filename, "content_type": file.content_type, "ocr_results": result}
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
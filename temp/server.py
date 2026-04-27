from fastapi import FastAPI, File, UploadFile
from ocr import do_ocr
import os
import tempfile

app = FastAPI()

@app.get("/")
def root():
    return {"status": "OCR server is running"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Call OCR function with file path
        result = do_ocr(tmp_path)
        return {"filename": file.filename, "content_type": file.content_type, "ocr_results": result}
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

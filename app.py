import io
import os
import sys
import torch
import asyncio
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.responses import Response
from fastapi.security import APIKeyHeader
import uvicorn

API_KEY = os.getenv('API_KEY')

if not API_KEY:
    print('api key required')
    sys.exit(1)

app = FastAPI()

def verify_key(api_key: str = Depends(APIKeyHeader(name='X-API-Key'))):
    if api_key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    return api_key

predictor = torch.hub.load('Stable-X/StableNormal', 'StableNormal', trust_repo=True)
predictor = predictor.to('cuda', dtype=torch.float16)

model_lock = asyncio.Lock()

def run_inference(img: Image.Image):
    with torch.inference_mode():
        return predictor(img)

@app.post("/normal/generate")
async def normal_generate(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_key)
):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert('RGB')
    except Exception:
        raise HTTPException(status_code=400, detail='invalid png')

    width, height = image.size
    if width > 1024 or height > 1024:
        raise HTTPException(
            status_code=400, 
            detail='image too large max (1024x1024)'
        )

    async with model_lock:
        try:
            normal_image = await asyncio.to_thread(run_inference, image)
        except Exception as e:
            print(e)
            raise HTTPException(status_code=500)

    output_buffer = io.BytesIO()
    normal_image.save(output_buffer, format='PNG')
    output_buffer.seek(0)

    return Response(content=output_buffer.getvalue(), media_type='image/png')

if __name__ == '__main__':
    uvicorn.run('app:app', host='0.0.0.0', port=8000, reload=False)
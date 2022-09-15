# import modules
from fastapi import FastAPI, Request
from starlette.responses import JSONResponse
import pandas as pd
import pickle
import os
import google.cloud.storage as gs
from server import LGBM

app = FastAPI()
model = LGBM()

# Health route
@app.get(os.environ['AIP_HEALTH_ROUTE'], status_code=200)
def health():
    return {"status": "healthy"}

# Predict route
@app.post(os.environ['AIP_PREDICT_ROUTE'])
async def predict(request: Request):
    body = await request.json()
    
    instances = body["instances"]
    output = []
    for i in instances:
        output.append(model.predict(i))
    return JSONResponse({"predictions": output})

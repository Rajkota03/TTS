from fastapi import FastAPI
from pydantic import BaseModel
from TTS.api import TTS
import base64

app = FastAPI()
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")

class Req(BaseModel):
    text: str

@app.post("/synthesize")
async def synth(req: Req):
    wav = tts.tts(req.text)
    b64 = base64.b64encode(wav).decode("utf-8")
    return {"audio": b64}

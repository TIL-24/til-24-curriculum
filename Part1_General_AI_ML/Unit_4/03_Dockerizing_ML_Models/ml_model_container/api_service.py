import json
from fastapi import FastAPI, Request
import base64
from ASRManager import ASRManager

app = FastAPI()

# instantiate model in the class here
asr_extractor = ASRManager()


@app.get("/health")
def health():
    return {"message": "health ok"}


# handle both post and get requests just in case
@app.post("/stt")
@app.get("/stt")
async def stt(request: Request):
    """
    Performs ASR given the filepath of an audio file
    Returns transcription of the audio
    """
    # for GCP batch prediction deployment
    # get base64 encoded string, convert back into bytes
    input_json = json.loads(await request.body())
    predictions = []
    for audio in input_json["instances"]:
        # each is a dict with one key "b64" and the value as a b64 encoded string
        audio_bytes = base64.b64decode(audio["b64"])
        transcription = asr_extractor.stt(audio_bytes)
        predictions.append(transcription)
    return {"predictions": predictions}

import asyncio
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import Body, FastAPI, Query, status
from fastapi.responses import StreamingResponse

from model_work import audio_Model, text_Model
from schemas import studentRequestModel, textRequestModel, textResponseModel

ml_models = {}


@asynccontextmanager
async def liefspan(app: FastAPI):

    print("[+] Start Loading models..... [+]")

    text_m_obj = text_Model()
    text_m_obj.load_pipeline()
    ml_models["text_m_obj"] = text_m_obj
    print("loaded text_m_obj")

    audio_m_obj = audio_Model()
    audio_m_obj.load_audio_model()
    ml_models["audio_m_obj"] = audio_m_obj
    print("loaded audio_m_obj")

    yield
    ml_models.clear()


app = FastAPI(lifespan=liefspan)


@app.get("/")
def home():
    """Home method"""
    return {"message": "Hello World"}


@app.post("/get_student")
def get_student(body: studentRequestModel = Body(...)) -> textResponseModel:
    """get method to get student data"""

    start_time = time.time()
    print(body)
    result = "OK"
    return textResponseModel(
        execution_time=int(time.time() - start_time), result=result
    )


@app.post("/sync")
def sync_prediction(prompt: str) -> textResponseModel:
    """post method to test sync method"""

    start_time = time.time()
    time.sleep(5)
    print("prompt : ", prompt)
    result = "OK"
    return textResponseModel(
        execution_time=int(time.time() - start_time), result=result
    )


@app.post("/async")
async def async_prediction() -> textResponseModel:
    """post method to test async method"""

    start_time = time.time()
    await asyncio.sleep(5)

    result = "OK"
    return textResponseModel(
        execution_time=int(time.time() - start_time), result=result
    )


@app.post("/text_gen")
def serve_text_gen(body: textRequestModel = Body(...)) -> textResponseModel:
    """post method for text_gen"""
    start_time = time.time()
    generated_text = ml_models["text_m_obj"].predict(user_message=body.prompt)

    return textResponseModel(
        execution_time=int(time.time() - start_time), result=generated_text
    )


@app.get(
    "/audio_gen",
    responses={status.HTTP_200_OK: {"content": {"audio/wav": {}}}},
    response_class=StreamingResponse,
)
async def serve_audio_gen(
    prompt=Query(...),
    prest: audio_Model.VoicePresets = Query(default="v2/en_speaker_9"),
) -> StreamingResponse:
    """async post method for audio_gen"""
    output_audio_array = ml_models["audio_m_obj"].generate_audio(prompt, prest)

    return StreamingResponse(
        output_audio_array,
        media_type="audio/wav",
        headers={"Content-Disposition": "inline; filename=generated_audio.wav"},
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8888, reload=True)

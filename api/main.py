from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

class EmailRequest(BaseModel):
    text: str

spam_classifier = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global spam_classifier


    model_id = "OliverArt5500/klasyfikatorspamu1"


    spam_classifier = pipeline("text-classification", model=model_id, tokenizer=model_id)
    print("Model gotowy")

    yield

    print("Wyłączanie API")
    spam_classifier = None


app = FastAPI(title="klasyfikator spamu", lifespan=lifespan)


@app.post("/predict")
async def predict_spam(request: EmailRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Email text cannot be empty.")

    try:
        prediction = spam_classifier(request.text, truncation=True, max_length=512)[0]

        return {
            "label": prediction["label"],
            "confidence_score": prediction["score"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

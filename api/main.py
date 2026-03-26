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
    print("Loading model from Hugging Face Hub...")


    model_id = "your-username/modern-phishing-detector"


    spam_classifier = pipeline("text-classification", model=model_id, tokenizer=model_id)
    print("Model loaded successfully. Ready for traffic!")

    yield

    print("Shutting down API and clearing memory.")
    spam_classifier = None


app = FastAPI(title="Modern Phishing Detection API", lifespan=lifespan)


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

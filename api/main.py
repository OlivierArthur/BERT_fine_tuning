from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="Klasyfikator SPAM API")

NAZWA = "OliverArt5500/klasyfikatorspamu1"

klasyfikator = pipeline("text-classification", model=NAZWA, tokenizer=NAZWA)

class Email(BaseModel):
    text: str

@app.post("/predict")
def predict(email: Email):
    wyniki = klasyfikator(email.text, truncation=True, max_length=512)
    return {"prediction": wyniki[0]}

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app=FastAPI()
sentiment_model=pipeline("sentiment-analysis")

class TextInput(BaseModel):
    text:str

@app.get("/health")
def health_check():
    return {"status":"ok"}
@app.post("/predict")
def predict_sentiment(input: TextInput):

    result = sentiment_model(input.text)
    return{"result":result}
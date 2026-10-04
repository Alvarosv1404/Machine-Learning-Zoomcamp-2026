import pickle
from fastapi import FastAPI
from pydantic import BaseModel

with open("pipeline_v1.bin", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

class Client(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

@app.post("/")
def score(client: Client):
    p = model.predict_proba(client.model_dump())[0][1]
    return {"probability": float(p)}
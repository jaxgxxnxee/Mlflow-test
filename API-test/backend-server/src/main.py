from fastapi import FastAPI, Request
from joblib import dump, load
import json
import pandas as pd

app = FastAPI()

model = load('s3/iris_classification.joblib') 
with open('s3/target_index.json', 'r') as f:
    target_index = json.load(f)


@app.get("/")
def read_root():
    return {"Hello": "World"}

def get_prediction(input_df: pd.DataFrame) -> list:
    x = input_df.values
    y_pred = model.predict(x)
    return y_pred.tolist()
    # return [{i: target_index[str(y)]} for i, y in enumerate(y_pred)]

@app.post("/predict")
async def predict(dataset: Request) -> str:
    dataset = await dataset.json()
    input_df = pd.DataFrame(dataset['data'], index=dataset['index'], columns=dataset['columns'])
    return {"predict": get_prediction(input_df)}
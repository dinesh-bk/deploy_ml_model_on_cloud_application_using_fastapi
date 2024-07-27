# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict
import pandas as pd
import joblib
from ml.data import process_data
from ml.model import inference

app = FastAPI()

# Load the model and encoder
model = joblib.load("./model/trained_model.pkl")
encoder = joblib.load("./model/encoder.pkl")
lb = joblib.load("./model/labelizer.pkl")

class CensusData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias='education-num')
    marital_status: str = Field(alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias='capital-gain')
    capital_loss: int = Field(alias='capital-loss')
    hours_per_week: int = Field(alias='hours-per-week')
    native_country: str = Field(alias='native-country')

    model_config = ConfigDict(
        #n example to the schema of our API
        json_schema_extra = {
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States"
            }
        }
    )
@app.get("/")
async def root():
    return {"message": "Welcome to the Census Income Prediction API!"}

@app.post("/predict")
async def predict(data: CensusData):
    # Convert input data to DataFrame
    df = pd.DataFrame([data.dict(by_alias=True)])

    # Process the input data
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X, _, _, _ = process_data(
        df, categorical_features=cat_features, training=False, encoder=encoder, lb=lb
    )

    # Make prediction
    prediction = inference(model, X)
    prediction_label = lb.inverse_transform(prediction)[0]
    print(f"Raw prediction: {prediction}, Label: {prediction_label}")
    return {"prediction": prediction_label}
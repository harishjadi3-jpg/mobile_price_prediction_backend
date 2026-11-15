
import json
import joblib
import numpy as np

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from tensorflow.keras.models import load_model

# Get the project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

# Load the trained ANN model
model = load_model(MODEL_DIR / "mobile_price_model.keras")

# Load the fitted StandardScaler
scaler = joblib.load(MODEL_DIR / "scaler.pkl")

# Load the correct feature order
with open(MODEL_DIR / "feature_columns.json", "r") as file:
    feature_columns = json.load(file)

# Create FastAPI application
app = FastAPI(title="Mobile Price Prediction API")

# Allow our React frontend to communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this to your frontend URL before deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the input features
class MobileFeatures(BaseModel):
    battery_power: int = Field(ge=1, le=2000)
    blue: int = Field(ge=0, le=1)
    clock_speed: float = Field(ge=0.1, le=3.0)
    dual_sim: int = Field(ge=0, le=1)
    fc: int = Field(ge=0, le=19)
    four_g: int = Field(ge=0, le=1)
    int_memory: int = Field(ge=2, le=64)
    m_dep: float = Field(ge=0.1, le=1.0)
    mobile_wt: int = Field(ge=80, le=200)
    n_cores: int = Field(ge=1, le=8)
    pc: int = Field(ge=0, le=20)
    px_height: int = Field(ge=0, le=1960)
    px_width: int = Field(ge=500, le=1998)
    ram: int = Field(ge=256, le=3998)
    sc_h: int = Field(ge=5, le=19)
    sc_w: int = Field(ge=0, le=18)
    talk_time: int = Field(ge=2, le=20)
    three_g: int = Field(ge=0, le=1)
    touch_screen: int = Field(ge=0, le=1)
    wifi: int = Field(ge=0, le=1)


@app.get("/")
def home():
    return {"message": "Mobile Price Prediction API is running!"}


@app.post("/predict")
def predict_price(data: MobileFeatures):

    # Convert input into a dictionary
    input_data = data.model_dump()

    # Arrange features in the same order used during training
    input_array = np.array(
        [[input_data[column] for column in feature_columns]],
        dtype=float
    )

    # Apply the previously fitted scaler
    scaled_data = scaler.transform(input_array)

    # Get ANN predictions
    probabilities = model.predict(scaled_data, verbose=0)

    predicted_class = int(np.argmax(probabilities, axis=1)[0])

    # Convert class into a readable category
    price_categories = {
        0: "Low Cost",
        1: "Medium Cost",
        2: "High Cost",
        3: "Very High Cost"
    }

    return {
        "predicted_class": predicted_class,
        "price_category": price_categories[predicted_class],
        "confidence": round(float(np.max(probabilities)) * 100, 2)
    }

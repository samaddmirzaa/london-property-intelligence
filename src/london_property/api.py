from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal

from london_property.model import (
    load_bundle, load_postcodes, find_postcode, predict_price
)

app = FastAPI(title='London Property Intelligence API')

bundle = load_bundle()
postcodes = load_postcodes()

class PropertyRequest(BaseModel):
    postcode: str = Field(examples=['E15 4EJ'])
    floor_area: float = Field(ge=15, le=500, examples=[70])
    property_type: Literal['Flat', 'Terraced', 'Semi-Detached', 'Detached']
    tenure: Literal['Freehold', 'Leasehold']
    built_year: int = Field(ge=1900, le=2024, examples=[1969])

class PriceResponse(BaseModel):
    predicted_price: float
    postcode: str
    district: str

@app.get("/health")
def health():
    return {'Status': 'OK'}

@app.post("/predict-price", response_model=PriceResponse)
def predict(request: PropertyRequest):
    row = find_postcode(postcodes, request.postcode)
    if row is None:
        raise HTTPException(status_code=404, detail='Postcode not found')

    price = predict_price(
        bundle, row,
        request.floor_area,
        request.property_type,
        request.tenure,
        request.built_year
    )
    return PriceResponse(
        predicted_price=round(price, 2),
        postcode=row['postcode'],
        district=row['postcode'].split(' ')[0]
    )


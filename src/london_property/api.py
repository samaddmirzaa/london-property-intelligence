import logging
from functools import cache
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from groq import GroqError, RateLimitError
from pydantic import BaseModel, Field

from london_property.agent import chat, make_llm, make_tools
from london_property.model import load_bundle, load_postcodes, find_postcode, predict_price
from london_property.tools import (
    PostcodeNotFound, load_area_data, get_area_profile, get_neighbourhood, find_nearby, get_guide, council_tax,
    stamp_duty
)

app = FastAPI(title='London Property Intelligence API')

bundle = load_bundle()
postcodes = load_postcodes()
area = load_area_data()
area_tools = make_tools(area)
log = logging.getLogger('uvicorn.error')


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


class Message(BaseModel):
    role: Literal['user', 'assistant']
    content: str = Field(min_length=1, max_length=2000)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1, max_length=20)


class ChatResponse(BaseModel):
    answer: str
    tools_used: list[dict]


@cache
def get_llm():
    """The LLM, created on first use, so the API and its tests can start without a Groq key."""
    return make_llm()


def run_tool(tool, *args):
    """Run an area tool, turning its errors into HTTP errors: unknown postcode -> 404, bad input -> 422."""
    try:
        return tool(*args)
    except PostcodeNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get('/')
def root():
    return {'Service': 'London Property Intelligence API', 'docs': '/docs'}


@app.get('/health')
def health():
    return {'Status': 'OK'}


@app.post('/predict-price', response_model=PriceResponse)
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


@app.get('/area/{postcode}')
def area_profile(postcode: str):
    """Location facts, ward profile and London averages for a postcode."""
    return run_tool(get_area_profile, area, postcode)


@app.get('/neighbourhood/{postcode}')
def neighbourhood(postcode: str):
    """One-call overview: location facts, ward and borough vs London, 10-minute walk, nearest essentials."""
    return run_tool(get_neighbourhood, area, postcode)


@app.get('/nearby/{postcode}')
def nearby(postcode: str,
           place_type: str = Query(alias='type', examples=['school']),
           radius_m: int = Query(1000, ge=50, le=3000),
           limit: int = Query(5, ge=1, le=20),
           subtype: str | None = Query(None, examples=['primary'])):
    """Places of one type near a postcode, nearest first."""
    return run_tool(find_nearby, area, postcode, place_type, radius_m, limit, subtype)


@app.get('/guide/{guide}')
def guide(guide: str, question: str = Query(min_length=1, max_length=200, examples=['service charges'])):
    """The sections of a GOV.UK buyer's guide that best match a question."""
    return run_tool(get_guide, area, guide, question)

@app.get('/council-tax/{postcode}')
def council_tax_bill(postcode: str, band: str = Query(examples=['C'])):
    """Annual council tax for a band (A-H) in the postcode's borough."""
    return run_tool(council_tax, area, postcode, band)


@app.get('/stamp-duty')
def stamp_duty_bill(price: int = Query(gt=0, le=100_000_000, examples=[500000]),
                    first_time_buyer: bool = False,
                    additional_property: bool = False,
                    non_uk_resident: bool = False):
    """Stamp Duty Land Tax on a residential purchase in England."""
    return run_tool(stamp_duty, price, first_time_buyer, additional_property, non_uk_resident)


@app.post('/chat', response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """Ask the area chatbot. Send the whole conversation each time: the API keeps no memory between requests."""
    if request.messages[-1].role != 'user':
        raise HTTPException(status_code=422, detail='The last message must be from the user')
    try:
        return chat([m.model_dump() for m in request.messages], area_tools, get_llm())
    except RateLimitError:
        raise HTTPException(status_code=429, detail='The chatbot is busy. Please try again in a minute.')
    except GroqError:
        log.exception('LLM call failed')
        raise HTTPException(status_code=503, detail='The chatbot is unavailable right now. Please try again later.')
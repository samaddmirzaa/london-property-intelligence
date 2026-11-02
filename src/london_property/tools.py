"""Area tools: the functions the chatbot calls to answer questions about a London postcode.

Every function returns plain Python data (dicts, lists, numbers, strings, None),
so its result can go straight back to the LLM as JSON or out through the API.
"""
import json
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from scipy.spatial import cKDTree

project_root = Path(__file__).resolve().parents[2]
area_dir = project_root/'data'/'processed'/'area'
london = 'E12000007'

place_types = ['school', 'gp', 'supermarket', 'park', 'listed building', 'restaurant', 'cafe', 'pub or bar',
               'gym', 'dentist', 'takeaway', 'bank', 'pharmacy', 'childcare', 'library', 'cinema', 'theatre']
max_radius_m = 3000

# Each council tax band is a fixed fraction of Band D, set in law
band_ratio = {'A': 6/9, 'B': 7/9, 'C': 8/9, 'D': 9/9, 'E': 11/9, 'F': 13/9, 'G': 15/9, 'H': 18/9}

class PostcodeNotFound(ValueError):
    """Raised when a postcode isn't a live London postcode (the API turns it into a 404)."""

@dataclass
class AreaData:
    """The three area tables plus a KD-tree over the points, loaded once at startup."""
    postcodes: pd.DataFrame
    profile: pd.DataFrame
    points: pd.DataFrame
    tree: cKDTree


def load_area_data():
    points = pd.read_parquet(area_dir/'points.parquet')
    return AreaData(
        postcodes=pd.read_parquet(area_dir/'postcodes.parquet').set_index('postcode'),
        profile=pd.read_parquet(area_dir/'ward_profile.parquet').set_index('ward_code'),
        points=points,
        tree=cKDTree(points[['easting', 'northing']].to_numpy()),
    )


def to_plain(obj):
    """A pandas row or table as plain Python: NaN becomes None, numpy numbers become Python numbers."""
    return json.loads(obj.to_json(orient='records') if isinstance(obj, pd.DataFrame) else obj.to_json())


def normalise_postcode(text):
    """'e83rl' or ' e8 3rl ' -> 'E8 3RL'. The last three characters are always the inward code."""
    compact = text.upper().replace(' ', '')
    return f'{compact[:-3]} {compact[-3:]}'


def lookup_postcode(data, postcode):
    """The postcode's row in the postcode table, or a clear error if it isn't a live London postcode."""
    key = normalise_postcode(postcode)
    if key not in data.postcodes.index:
        raise PostcodeNotFound(f'{postcode} is not a live London postcode')
    return key, data.postcodes.loc[key]


def get_area_profile(data, postcode):
    """Everything known about a postcode's exact location and its ward, with London averages to compare."""
    key, pc = lookup_postcode(data, postcode)
    return {
        'postcode': key,
        'location': to_plain(pc),
        'ward': to_plain(data.profile.loc[pc['ward_code']]),
        'london_average': to_plain(data.profile.loc[london].drop(['ward_name', 'borough_code', 'borough_name'])),
    }


def find_nearby(data, postcode, place_type, radius_m=1000, limit=5, subtype=None):
    """Places of one type within a radius of a postcode, nearest first, with a total count."""
    if place_type not in place_types:
        raise ValueError(f"Unknown place type '{place_type}'. Choose from: {', '.join(place_types)}")
    key, pc = lookup_postcode(data, postcode)
    radius_m = min(radius_m, max_radius_m)

    near = data.points.iloc[data.tree.query_ball_point([pc['easting'], pc['northing']], r=radius_m)]
    near = near[near['type'] == place_type]
    if subtype:
        near = near[near['subtype'].str.contains(subtype, case=False, regex=False)]
    distance = ((near['easting'] - pc['easting']) ** 2 + (near['northing'] - pc['northing']) ** 2) ** 0.5
    near = near.assign(distance_m=distance.round().astype(int)).sort_values('distance_m').drop_duplicates('id')

    return {
        'postcode': key, 'type': place_type, 'subtype': subtype, 'radius_m': radius_m,
        'count': len(near),
        'nearest': to_plain(near.head(limit)[['name', 'subtype', 'info', 'distance_m', 'lat', 'lon']]),
    }


def council_tax(data, postcode, band):
    """Annual council tax for a band (A-H) in the postcode's borough, 2026-27."""
    band = band.strip().upper()
    if band not in band_ratio:
        raise ValueError('Council tax band must be a letter from A to H')
    key, pc = lookup_postcode(data, postcode)
    band_d = float(data.profile.loc[pc['ward_code'], 'council_tax_band_d'])
    return {
        'postcode': key, 'borough': pc['borough_name'], 'band': band, 'year': '2026-27',
        'annual_bill': round(band_d * band_ratio[band], 2), 'band_d_bill': band_d,
    }


def stamp_duty(price, first_time_buyer=False, additional_property=False, non_uk_resident=False):
    """Stamp Duty Land Tax on a residential purchase in England (rates from 1 April 2025)."""
    if first_time_buyer and additional_property:
        raise ValueError('A first-time buyer cannot already own another property')
    if first_time_buyer and price <= 500_000:
        bands = [(300_000, 0.00), (500_000, 0.05)]
    else:
        bands = [(125_000, 0.00), (250_000, 0.02), (925_000, 0.05), (1_500_000, 0.10), (math.inf, 0.12)]
    surcharge = 0.05 * additional_property + 0.02 * non_uk_resident

    tax, lower = 0.0, 0
    for upper, rate in bands:
        if price > lower:
            tax += (min(price, upper) - lower) * (rate + surcharge)
        lower = upper

    tax = math.floor(tax)
    return {
        'price': price, 'stamp_duty': tax, 'effective_rate_pct': round(tax / price * 100, 2),
        'first_time_buyer_relief': first_time_buyer and price <= 500_000,
    }
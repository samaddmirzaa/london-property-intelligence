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
walk_m = 800    # a 10-minute walk at about 80 metres a minute

# Ward measures compared with every other London ward, and borough measures compared with London
rank_cols = ['crime_total', 'crime_burglary', 'crime_robbery', 'crime_theft', 'crime_violence_against_the_person',
             'crime_vehicle_offences', 'crime_public_order_offences', 'crime_arson_and_criminal_damage',
             'crime_drug_offences', 'crime_sexual_offences', 'crime_possession_of_weapons',
             'crime_miscellaneous_crimes_against_society',
             'pct_age_0_14', 'pct_age_15_29', 'pct_age_30_44', 'pct_age_45_64', 'pct_age_65_plus',
             'pct_owned', 'pct_social_rented', 'pct_private_rented', 'ptal_ai', 'no2', 'pm25', 'pm10']
borough_cols = ['price_avg', 'price_change_12m', 'price_flat', 'price_terraced', 'price_semi', 'price_detached',
                'rent_avg', 'rent_change_12m', 'rent_1bed', 'rent_2bed', 'rent_3bed', 'rent_4bed_plus',
                'council_tax_band_d']

# The nearest one of each everyday essential: label -> (type, subtype)
essentials = {'primary school': ('school', 'primary'), 'secondary school': ('school', 'secondary'),
              'gp': ('gp', None), 'supermarket': ('supermarket', None), 'park': ('park', 'park'),
              'pharmacy': ('pharmacy', None)}

# Each council tax band is a fixed fraction of Band D, set in law
band_ratio = {'A': 6/9, 'B': 7/9, 'C': 8/9, 'D': 9/9, 'E': 11/9, 'F': 13/9, 'G': 15/9, 'H': 18/9}


class PostcodeNotFound(ValueError):
    """Raised when a postcode isn't a live London postcode (the API turns it into a 404)."""


@dataclass
class AreaData:
    """The area tables, a KD-tree over the points and the walk benchmark, loaded once at startup."""
    postcodes: pd.DataFrame
    profile: pd.DataFrame
    points: pd.DataFrame
    tree: cKDTree
    walk_benchmark: pd.DataFrame


def load_area_data():
    points = pd.read_parquet(area_dir/'points.parquet')
    return AreaData(
        postcodes=pd.read_parquet(area_dir/'postcodes.parquet').set_index('postcode'),
        profile=pd.read_parquet(area_dir/'ward_profile.parquet').set_index('ward_code'),
        points=points,
        tree=cKDTree(points[['easting', 'northing']].to_numpy()),
        walk_benchmark=pd.read_parquet(area_dir/'walk_benchmark.parquet'),
    )


def to_plain(obj, orient='index'):
    """A pandas row or table as plain Python: NaN becomes None, numpy numbers become Python numbers."""
    return json.loads(obj.to_json(orient=orient))


def count_by_type(points, tree, easting, northing, radius_m):
    """How many places of each type lie within radius_m (a park with several gates counts once)."""
    near = points.iloc[tree.query_ball_point([easting, northing], r=radius_m)]
    return near.drop_duplicates(['type', 'id'])['type'].value_counts().reindex(place_types, fill_value=0)


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
        'nearest': to_plain(near.head(limit)[['name', 'subtype', 'info', 'distance_m', 'lat', 'lon']], orient='records'),
    }


def get_neighbourhood(data, postcode):
    """A one-call overview of a postcode for the chatbot: location facts, the ward and borough compared
    with London, what is within a 10-minute walk compared with all of London, and the nearest essentials."""
    key, pc = lookup_postcode(data, postcode)
    ward = data.profile.loc[pc['ward_code']]
    ldn = data.profile.loc[london]
    others = data.profile.drop(index=london)

    higher = (others[rank_cols] < ward[rank_cols].astype(float)).mean().mul(100).round()
    higher = higher.where(ward[rank_cols].notna()).astype('Int64')
    ward_vs_london = pd.DataFrame({'ward': ward[rank_cols], 'london': ldn[rank_cols], 'higher_than_pct_of_wards': higher})
    borough_vs_london = pd.DataFrame({'borough': ward[borough_cols], 'london': ldn[borough_cols]})

    counts = count_by_type(data.points, data.tree, pc['easting'], pc['northing'], walk_m)
    beats = (data.walk_benchmark < counts).mean().mul(100).round().astype(int)
    walk = pd.DataFrame({'count': counts, 'more_than_pct_of_london_postcodes': beats})

    nearest = {}
    for label, (place_type, subtype) in essentials.items():
        found = find_nearby(data, key, place_type, max_radius_m, 1, subtype)['nearest']
        nearest[label] = {k: found[0][k] for k in ('name', 'distance_m', 'info')} if found else None

    return {
        'postcode': key,
        'location': to_plain(pc.drop(['lat', 'lon', 'easting', 'northing', 'lsoa_code', 'ward_code', 'borough_code'])),
        'ward': to_plain(ward[['ward_name', 'borough_name', 'population', 'households', 'ptal_band',
                               'crime_months', 'price_month', 'rent_month']]),
        'ward_vs_london': to_plain(ward_vs_london),
        'borough_vs_london': to_plain(borough_vs_london),
        'within_10_min_walk': to_plain(walk),
        'nearest': nearest,
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
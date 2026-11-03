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
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from sklearn.feature_extraction.text import TfidfVectorizer

project_root = Path(__file__).resolve().parents[2]
area_dir = project_root/'data'/'processed'/'area'
london = 'E12000007'

place_types = ['school', 'gp', 'supermarket', 'park', 'listed building', 'restaurant', 'cafe', 'pub or bar',
               'gym', 'dentist', 'takeaway', 'bank', 'pharmacy', 'childcare', 'library', 'cinema', 'theatre']
max_radius_m = 3000
walk_m = 800    # a 10-minute walk at about 80 metres a minute

# Ward measures compared with every other London ward, and borough measures compared with London
# The crime breakdown is ranked too, but kept out of the high/low lists so that 11 crime types can't
# crowd out everything else.
crime_types = ['crime_burglary', 'crime_robbery', 'crime_theft', 'crime_violence_against_the_person',
               'crime_vehicle_offences', 'crime_public_order_offences', 'crime_arson_and_criminal_damage',
               'crime_drug_offences', 'crime_sexual_offences', 'crime_possession_of_weapons',
               'crime_miscellaneous_crimes_against_society']
rank_cols = ['crime_total', *crime_types,
             'pct_age_0_14', 'pct_age_15_29', 'pct_age_30_44', 'pct_age_45_64', 'pct_age_65_plus',
             'pct_owned', 'pct_social_rented', 'pct_private_rented']
# Transport access and air quality at the exact postcode, compared with every London postcode
location_cols = ['ptal_ai', 'no2', 'pm25', 'pm10']
borough_cols = ['price_avg', 'price_change_12m', 'price_flat', 'price_terraced', 'price_semi', 'price_detached',
                'rent_avg', 'rent_change_12m', 'rent_1bed', 'rent_2bed', 'rent_3bed', 'rent_4bed_plus',
                'council_tax_band_d']

# The nearest one of each everyday essential: label -> (type, subtype)
essentials = {'primary school': ('school', 'primary'), 'secondary school': ('school', 'secondary'),
              'gp': ('gp', None), 'supermarket': ('supermarket', None), 'park': ('park', 'park'),
              'pharmacy': ('pharmacy', None)}

# The GOV.UK guides the chatbot can quote (their page addresses on gov.uk)
guide_slugs = ['buying-a-home', 'stamp-duty-land-tax', 'leasehold-property', 'joint-property-ownership',
               'shared-ownership-scheme', 'first-homes-scheme', 'lifetime-isa', 'right-to-buy-buying-your-council-home',
               'council-tax', 'get-information-about-property-and-land', 'planning-permission-england-wales',
               'party-walls-building-works']

# Each council tax band is a fixed fraction of Band D, set in law
band_ratio = {'A': 6/9, 'B': 7/9, 'C': 8/9, 'D': 9/9, 'E': 11/9, 'F': 13/9, 'G': 15/9, 'H': 18/9}


class PostcodeNotFound(ValueError):
    """Raised when a postcode isn't a live London postcode (the API turns it into a 404)."""


@dataclass
class AreaData:
    """Everything the tools read, loaded once at startup: the area tables, a KD-tree over the points,
    the walk benchmark, and the GOV.UK guide sections with their TF-IDF search index."""
    postcodes: pd.DataFrame
    profile: pd.DataFrame
    points: pd.DataFrame
    tree: cKDTree
    walk_benchmark: pd.DataFrame
    guides: pd.DataFrame
    guide_vectorizer: TfidfVectorizer
    guide_matrix: csr_matrix


def load_area_data():
    points = pd.read_parquet(area_dir/'points.parquet')
    guides = pd.read_parquet(area_dir/'guides.parquet')
    vectorizer = TfidfVectorizer(stop_words='english', sublinear_tf=True)
    documents = (guides['guide_title'] + ' ' + guides['part_title'] + ' ') * 2 + guides['text']
    return AreaData(
        postcodes=pd.read_parquet(area_dir/'postcodes.parquet').set_index('postcode'),
        profile=pd.read_parquet(area_dir/'ward_profile.parquet').set_index('ward_code'),
        points=points,
        tree=cKDTree(points[['easting', 'northing']].to_numpy()),
        walk_benchmark=pd.read_parquet(area_dir/'walk_benchmark.parquet'),
        guides=guides,
        guide_vectorizer=vectorizer,
        guide_matrix=vectorizer.fit_transform(documents),
    )

def to_plain(obj, orient='index'):
    """A pandas row or table as plain Python: NaN becomes None, numpy numbers become Python numbers."""
    return json.loads(obj.to_json(orient=orient))


def count_by_type(points, tree, easting, northing, radius_m):
    """How many places of each type lie within radius_m (a park with several gates counts once)."""
    near = points.iloc[tree.query_ball_point([easting, northing], r=radius_m)]
    return near.drop_duplicates(['type', 'id'])['type'].value_counts().reindex(place_types, fill_value=0)

def london_percentile(table, row):
    """Where each value in row sits within the same column of table, from 0 to 100 (50 = typical).
    Equal values count half, so a common value such as 0 cinemas lands mid-table; missing values are ignored."""
    below = table.lt(row).sum() + table.eq(row).sum() / 2
    return (below / table.count()).mul(100).round().where(row.notna()).astype('Int64')

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
    """A one-call overview of a postcode for the chatbot: location facts, what is unusually high or low,
    the postcode, ward and borough compared with London, what is within a 10-minute walk compared with
    all of London, and the nearest essentials."""
    key, pc = lookup_postcode(data, postcode)
    ward = data.profile.loc[pc['ward_code']]
    ldn = data.profile.loc[london]
    others = data.profile.drop(index=london)

    ward_values = ward[rank_cols].astype(float)
    ward_pct = london_percentile(others[rank_cols], ward_values)
    ward_vs_london = pd.DataFrame({'ward': ward_values, 'london': ldn[rank_cols], 'london_percentile': ward_pct})

    borough, ldn_borough = ward[borough_cols].astype(float), ldn[borough_cols].astype(float)
    diff_pct = (borough / ldn_borough - 1).mul(100).round().drop(['price_change_12m', 'rent_change_12m'])
    borough_vs_london = pd.DataFrame({'borough': borough, 'london': ldn_borough, 'diff_pct': diff_pct})

    here = pc[location_cols].astype(float)
    here_pct = london_percentile(data.postcodes[location_cols], here)
    here_vs_london = pd.DataFrame({'here': here, 'london': ldn[location_cols], 'london_percentile': here_pct})

    counts = count_by_type(data.points, data.tree, pc['easting'], pc['northing'], walk_m)
    walk_pct = london_percentile(data.walk_benchmark, counts)
    walk = pd.DataFrame({'count': counts, 'london_percentile': walk_pct})

    ranks = pd.concat([ward_pct.drop(crime_types).rename(lambda c: f'{c} (ward)'),
                       here_pct.rename(lambda c: f'{c} (this postcode)'),
                       walk_pct.rename(lambda c: f'{c} (10-min walk)')]).dropna()
    high = ranks[ranks >= 80].sort_values(ascending=False).head(8)
    low = ranks[ranks <= 20].sort_values().head(8)

    # Coordinates and codes dropped (the LLM doesn't need them); station distance in metres like every other
    location = pc.drop(['lat', 'lon', 'easting', 'northing', 'lsoa_code', 'ward_code', 'borough_code', *location_cols])
    location = location.rename({'station_km': 'station_m'})
    location['station_m'] = round(location['station_m'] * 1000)

    nearest = {}
    for label, (place_type, subtype) in essentials.items():
        found = find_nearby(data, key, place_type, max_radius_m, 1, subtype)['nearest']
        nearest[label] = {k: found[0][k] for k in ('name', 'distance_m', 'info')} if found else None

    return {
        'postcode': key,
        'location': to_plain(location),
        'ward': to_plain(ward[['ward_name', 'borough_name', 'population', 'households',
                               'crime_months', 'price_month', 'rent_month']]),
        'higher_than_most_of_london': to_plain(high),
        'lower_than_most_of_london': to_plain(low),
        'here_vs_london': to_plain(here_vs_london),
        'ward_vs_london': to_plain(ward_vs_london),
        'borough_vs_london': to_plain(borough_vs_london),
        'within_10_min_walk': to_plain(walk),
        'nearest': nearest,
    }

def get_guide(data, guide, question):
    """The two sections of a GOV.UK guide that best match a question (TF-IDF similarity), with their links,
    plus the titles of the guide's other sections."""
    if guide not in guide_slugs:
        raise ValueError(f"Unknown guide '{guide}'. Choose from: {', '.join(guide_slugs)}")
    parts = data.guides[data.guides['guide'] == guide]
    query = data.guide_vectorizer.transform([question])
    scores = (data.guide_matrix[parts.index.to_numpy()] @ query.T).toarray().ravel()
    ranked = parts.iloc[(-scores).argsort(kind='stable')]
    return {
        'guide': ranked['guide_title'].iloc[0], 'updated': ranked['updated'].iloc[0],
        'sections': to_plain(ranked.head(2)[['part_title', 'url', 'text']], orient='records'),
        'other_sections': ranked['part_title'].iloc[2:].tolist(),
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

    tax, lower, breakdown = 0.0, 0, []
    for upper, rate in bands:
        if price > lower:
            band_tax = (min(price, upper) - lower) * (rate + surcharge)
            breakdown.append({'from': lower, 'to': min(price, upper), 'amount': min(price, upper) - lower,
                              'rate_pct': round((rate + surcharge) * 100), 'tax': round(band_tax)})
            tax += band_tax
        lower = upper

    tax = math.floor(tax)
    return {
        'price': price, 'stamp_duty': tax, 'effective_rate_pct': round(tax / price * 100, 2),
        'first_time_buyer_relief': first_time_buyer and price <= 500_000,
        'rates_from': '1 April 2025', 'bands': breakdown,
    }
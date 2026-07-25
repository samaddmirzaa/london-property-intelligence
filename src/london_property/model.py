import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Paths for the files
project_root = Path(__file__).resolve().parents[2]
model_path = project_root /'models'/'price_model.joblib'
postcode_path = project_root/'data'/'supplementary'/'postcodes_london.parquet'

# Defining the Option Strings
property_types = ['Flat', 'Terraced', 'Semi-Detached', 'Detached']
tenures = ['Freehold', 'Leasehold']

# Loaders for the Model and Postcode reference
def load_bundle():
    return joblib.load(model_path)

def load_postcodes():
    return pd.read_parquet(postcode_path)

# Preprocessing artifacts saved at training time reused so the app matches training exactly
def build_features(bundle, pc_row, floor_area, property_type, tenure, built_year):
    medians = bundle['impute_medians']
    encoding = bundle['target_encoding']
    global_mean = bundle['te_global_mean']
    district = pc_row['postcode'].split(' ')[0]

    values = {
        'log_tfarea': np.log10(floor_area),
        'numberrooms': medians['numberrooms'],
        'age_year': float(built_year),
        'lat': pc_row['lat'],
        'lon': pc_row['lon'],
        'zone': pc_row['zone'],
        'imd_index': pc_row['imd_index'],
        'log_income': np.log10(pc_row['avg_income']),
        'dist_station': pc_row['dist_station'],
        'propertytype_D': int(property_type == 'Detached'),
        'propertytype_F': int(property_type == 'Flat'),
        'propertytype_S': int(property_type == 'Semi-Detached'),
        'propertytype_T': int(property_type == 'Terraced'),
        'duration_F': int(tenure == 'Freehold'),
        'duration_L': int(tenure == 'Leasehold'),
        'year': 2024,
        'month': 6,
        'district_te': encoding.get(district, global_mean),
    }
    # One row DataFrame, columns reordered to the model's exact expected order
    return pd.DataFrame([values])[list(bundle['features'])]

def predict_price(bundle, pc_row, floor_area, property_type, tenure, built_year):
    X = build_features(bundle, pc_row, floor_area, property_type, tenure, built_year)
    return float(10 ** bundle['model'].predict(X)[0])

def find_postcode(postcodes, postcode):
    match = postcodes[postcodes['postcode'] == postcode.upper().strip()]
    if match.empty:
        return None
    return match.iloc[0]


def nearest_postcode(postcodes, lat, lon):
    d = (postcodes['lat'] - lat) ** 2 + (postcodes['lon'] - lon) ** 2
    return postcodes.loc[d.idxmin()]


from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pandas as pd
import seaborn as sns
from data_info import *
pd.set_option("display.precision", 2)
df = pd.read_csv(r"C:\Users\KIIT\Documents\coding\dengAI\raw.githubusercontent.com_burnpiro_dengai-predicting-disease-spread_master_dengue_features_train.csv")
df.describe()

LABEL_COLUMN = 'total_cases'
NUMERIC_COLUMNS = ['year',
                   'weekofyear',
                   'ndvi_ne',
                   'ndvi_nw',
                   'ndvi_se',
                   'ndvi_sw',
                   'precipitation_amt_mm',
                   'reanalysis_air_temp_k',
                   'reanalysis_avg_temp_k',
                   'reanalysis_dew_point_temp_k',
                   'reanalysis_max_air_temp_k',
                   'reanalysis_min_air_temp_k',
                   'reanalysis_precip_amt_kg_per_m2',
                   'reanalysis_relative_humidity_percent',
                   'reanalysis_sat_precip_amt_mm',
                   'reanalysis_specific_humidity_g_per_kg',
                   'reanalysis_tdtr_k',
                   'station_avg_temp_c',
                   'station_diur_temp_rng_c',
                   'station_max_temp_c',
                   'station_min_temp_c',
                   'station_precip_mm']
CATEGORICAL_COLUMNS = ['city']
CSV_COLUMNS = [LABEL_COLUMN] + CATEGORICAL_COLUMNS + NUMERIC_COLUMNS
CSV_COLUMNS_NO_LABEL = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS
CATEGORIES = {
    'city': ['sj', 'iq']
}
cols_to_norm = ['precipitation_amt_mm',
                'reanalysis_air_temp_k',
                'reanalysis_avg_temp_k',
                'reanalysis_dew_point_temp_k',
                'reanalysis_max_air_temp_k',
                'reanalysis_min_air_temp_k',
                'reanalysis_precip_amt_kg_per_m2',
                'reanalysis_relative_humidity_percent',
                'reanalysis_sat_precip_amt_mm',
                'reanalysis_specific_humidity_g_per_kg',
                'reanalysis_tdtr_k',
                'station_avg_temp_c',
                'station_diur_temp_rng_c',
                'station_max_temp_c',
                'station_min_temp_c',
                'station_precip_mm']
cols_to_scale = ['year',
                 'weekofyear']

TRAIN_DATASET_FRAC = 0.8
DATETIME_COLUMN = "week_start_date"
train_file = r"C:\Users\KIIT\Documents\coding\dengAI\raw.githubusercontent.com_burnpiro_dengai-predicting-disease-spread_master_dengue_features_train.csv"
test_file_in = r"C:\Users\KIIT\Documents\coding\dengAI\raw.githubusercontent.com_burnpiro_dengai-predicting-disease-spread_master_dengue_features_test.csv"
test_file = r"C:\Users\KIIT\Documents\coding\dengAI\raw.githubusercontent.com_burnpiro_dengai-predicting-disease-spread_master_dengue_features_test.csv"

def extract_data(train_file_path, columns, categorical_columns=CATEGORICAL_COLUMNS, categories_desc=CATEGORIES,
                 interpolate=True):
    # Read csv file and return
    all_data = pd.read_csv(train_file_path, usecols=columns)
    if categorical_columns is not None:
        # map categorical to columns
        for feature_name in categorical_columns:
            mapping_dict = {categories_desc[feature_name][i]: categories_desc[feature_name][i] for i in
                            range(0, len(categories_desc[feature_name]))}
            all_data[feature_name] = all_data[feature_name].map(mapping_dict)
        # Change mapped categorical data to 0/1 columns
        all_data = pd.get_dummies(all_data, prefix='', prefix_sep='')
    # fix missing data
    if interpolate:
        all_data = all_data.interpolate(method='linear', limit_direction='forward')
    return all_data

train_data = extract_data(train_file, CSV_COLUMNS)
# Get data for "sj" city and drop both binary columns
sj_train = train_data[train_data['sj'] == 1].drop(['sj', 'iq'], axis=1)
# Generate heatmap
corr = sj_train.corr()
mask = np.triu(np.ones_like(corr, dtype=np.bool))
plt.figure(figsize=(20, 10))
ax = sns.heatmap(
    corr, 
    mask=mask, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_title('Data correlation for city "sj"')
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
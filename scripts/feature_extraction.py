import dask.dataframe as dd
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, date, timedelta
from sklearn.preprocessing import MinMaxScaler
import holidays
from subprocess import call

# Defining attributes' names
header = [
    'time_key',                     # Date of the transaction (yyyymmdd)
    'location_key',                 # Shop location id
    'transaction_id',               # Id of the transaction
    'card_key',                     # Id of the client
    'ss_register_time_key',         # Time of first register of the client (yyyymmdd)
    'shop_trip_start_hour_key',     # Start hour of the shop trip (hmmss)
    'shop_trip_end_hour_key',       # End hour of the shop trip (hmmss)
    'transaction_end_hour_key',     # End hour of the transaction (payment) (hmmss)
    'rescan_fl',                    # 0: transaction not rescanned, 1: transaction rescanned
    'rescan_tp',                    # Type of the rescan: NONE, PARTIAL, COMPLETE
    'rescan_start_hour_key',        # Rescan start time of transaction (hmmss)
    'rescan_end_hour_key',          # Rescan end hour of product or transaction
    'transactional_total_amt_eur',  # Total amount of the transaction in euros
    'transactional_total_qt',       # Total amount of products in transaction
    'total_divergence_fl',          # 0: no divergence, 1: divergence
    'total_divergence_sign',        # =: same amount, -: scanned extra items +: scanned less items
    'total_divergence_amt_eur',     # Amount of divergence in euros
    'total_divergence_qt',          # Amount of divergent products
    'ean_key',                      # id of product (shared by same products) 
    'product_added_times_qt',       # Amount of times the product was scanned by the customer
    'product_removed_qt',           # Amount of times the product was removed by the customer
    'unknown_product_fl',           # 1: scanner failed to identify product, 0: scanner didn't fail 
    'rescan_product_fl',            # 0: product wasn't rescanned, 1: product was rescanned
    'product_total_amt_eur',        # Total amount of the product in euros (actually scanned by the customer)
    'product_total_qt',             # Total amount of the product (actually scanned by the customer)
    'product_divergence_fl',        # 0: product was correctly scanned, 1: product wasn't correctly scanned
    'product_divergence_amt_eur',   # Total amount of the divergence in euros
    'product_divergence_qt',        # Amount of products in divergence
    ]

# Defining columns' types
types = {
    'ean_key': 'object',
    'transaction_end_hour_key': 'object',
    'card_key': 'object',
    'rescan_start_hour_key': 'str',
    'rescan_end_hour_key': 'str',
    'shop_trip_start_hour_key': 'str',
    'shop_trip_end_hour_key': 'str'
}

########################
### DATA PREPARATION ###
########################

dates_to_parse = [
    'time_key',
    'ss_register_time_key'
]

# Reading the raw (preprocessed) data set
original_ddf = dd.read_csv(
    'data/processed/data_all_shops.csv',
    sep = ';',
    names = header,
    parse_dates = dates_to_parse,
    dtype = types
)

'''
################################
### PRODUCT DIVERGENCE SCORE ###
################################

In order to obtain a score for "how divergent" a product is (in our data), a product divergence score has been created.
Grouping the data by product and dividing the total product divergence by the frequency of the product wouldn't work because:

prod_div_score = product_total_divs / product_total_freq
               = 1 / 1 = 1 for product A,
        While: = 90 / 100 = 0.9 for product B

Product A's divergence score is higher than B's, although B's statistics are much more relevant.

Products' frequency varies between 1 and 100k.
Several (product) frequency intervals have been defined  ([0, 500[), [501, 1000[, ..., [99500, 100000[)
For the base interval ([0, 500]) a base weight has been defined (2) and it grows step times as the invervals grow, 
where step is the increment between intervals

        prod_div_score = product_total_divs * weight / product_total_freq,
where:  weight = base_weight + step * #_interval 

The default settings are:
    - base_weight = 2
    - step = 0.015
    - interval = 500
Different settings produce different results
'''

# Grouping by product (so we can compute the frequency of each unique product and amount of divergence occurrences)
def agg_fn(x):
     return pd.Series(dict(
                        prod_freq = x['ean_key'].count(),
                        prod_div_freq = x['product_divergence_fl'].sum()
                        )
                    )
prod_ddf = original_ddf.groupby(['ean_key']).apply(agg_fn) # Each product is an index

# Computing the weight for each unique product based on its' frequency
def prod_weight(freq, base_weight, step, interval):
    max_prod_freq = 110000
    for i in range(0, max_prod_freq, interval):
        if (freq >= i and freq < i + interval):
            return round(base_weight, 3)
        base_weight += step

# Computing the divergence score for each product
div_score_fn = lambda x: x['prod_div_freq'] * prod_weight(x['prod_freq'], 2, 0.015, 500) / x['prod_freq']
prod_ddf['prod_div_score'] = prod_ddf.apply(div_score_fn, axis = 1)

# Assigning score to products (weight matrix converted to dictionary for faster access/computation)
prod_ddf_dic = prod_ddf.prod_div_score.compute().to_dict()
# Divergence score = -1 fir EAN keys not available in the dictionary
score_fn = lambda x: prod_ddf_dic.get(x, -1)
original_ddf['prod_div_score'] = original_ddf['ean_key'].apply(score_fn)

# Grouping data by transaction
# Each transaction has a date, client and start/end shopping hour
atts_to_group = ['time_key', 'card_key', 'shop_trip_start_hour_key', 'shop_trip_end_hour_key']
def agg_fn(x):
     return pd.Series(dict(
                        # --- v1 data set attributes ---
                        # For each transation the values below don't change so it doesn't matter whether it is a min or max
                        ss_register_time_key = x['ss_register_time_key'].max(), 
                        transactional_total_amt_eur = x['transactional_total_amt_eur'].max(), 
                        transactional_total_qt = x['transactional_total_qt'].max(),
                        total_divergence_fl = x['total_divergence_fl'].max(),
                        unknown_product_fl = x['unknown_product_fl'].max(),

                        # --- v3 data set attributes ---
                        # Total scanned products
                        product_added_times_qt = x['product_added_times_qt'].sum(),
                        # Total removed scanned products
                        product_removed_qt = x['product_removed_qt'].sum(),
                        # Total diversity of products
                        number_different_products = x['ean_key'].count(),

                        # --- v4 data set attributes ---
                        transactional_div_score = (x['product_total_qt'] * x['prod_div_score']).sum()
                        )
                    )
ss_ddf_v1 = original_ddf.groupby(atts_to_group).apply(agg_fn).reset_index()

#{ echo "time_key;card_key;shop_trip_start_hour_key;shop_trip_end_hour_key;ss_register_time_key;transactional_total_amt_eur;transactional_total_qt;total_divergence_fl;unknown_product_fl"; cat ss_v1.csv; } > ss_v1.1.csv
#{ echo "transactional_total_amt_eur;transactional_total_qt;total_divergence_fl;unknown_product_fl;day;week_day;is_weekend;year;year_day;week_of_year;is_quarter_end;is_quarter_start;is_year_end;is_year_start;is_leap_year;month;week_of_month;first_month_half;second_month_half;is_month_start;is_month_end;month_days;shop_started_morning;shop_started_evening;shop_started_night;shop_ended_morning;shop_ended_evening;shop_ended_night;elapsed_register_time;total_shopping_time;mean_time_per_product"; cat ss_v2.csv; } > ss_v2.1.csv

'''
### DATASET V1 ###
At this point, the data set is grouped by transaction.
The baseline statistics have been computed on this version, using the following attributes
    - ss_register_time_key
    - transactional_total_amt_eur
    - transactional_total_qt
    - total_divergence_fl
    - unknown_product_fl
'''

#ss_ddf_v1.to_csv('/home/routar/FEUP/workspace/SONAE/ss_ml/newdata/remod/ss_filt_data_v1-*.csv', sep=';', header=False, index=False)

# Keeping v1 data set unchanged
ss_ddf_v2 = ss_ddf_v1

'''
######################
# FEATURE EXTRACTION #
######################
From now on, we are ready to extract features from the data. They are listed below
'''

# Converting quantities (float) to integers
ss_ddf_v2.transactional_total_qt = ss_ddf_v2.transactional_total_qt.astype(int)

### TIME PARSING ###

# Daily statistics
ss_ddf_v2['day'] = ss_ddf_v2.time_key.apply(lambda x: x.day) 
ss_ddf_v2['week_day'] = ss_ddf_v2.time_key.apply(lambda x: x.dayofweek)
ss_ddf_v2['is_weekend'] = ss_ddf_v2.week_day.apply(lambda x: 1 if x == 5 or x == 6 else 0)

# Yearly statistics
ss_ddf_v2['year'] = ss_ddf_v2.time_key.apply(lambda x: x.year) 
ss_ddf_v2['year_day'] = ss_ddf_v2.time_key.apply(lambda x: x.dayofyear)
ss_ddf_v2['week_of_year'] = ss_ddf_v2.time_key.apply(lambda x: x.weekofyear)
is_quarter_end = ss_ddf_v2.time_key.apply(lambda x: x.is_quarter_end) 
ss_ddf_v2['is_quarter_end'] = is_quarter_end.apply(lambda x: 1 if x == True else 0)
is_quarter_start = ss_ddf_v2.time_key.apply(lambda x: x.is_quarter_start) 
ss_ddf_v2['is_quarter_start'] = is_quarter_start.apply(lambda x: 1 if x == True else 0) 
is_year_end = ss_ddf_v2.time_key.apply(lambda x: x.is_year_end) 
ss_ddf_v2['is_year_end'] = is_year_end.apply(lambda x: 1 if x == True else 0) 
is_year_start = ss_ddf_v2.time_key.apply(lambda x: x.is_year_start)
ss_ddf_v2['is_year_start'] = is_year_start.apply(lambda x: 1 if x == True else 0) 
is_leap_year = ss_ddf_v2.time_key.apply(lambda x: x.is_leap_year) 
ss_ddf_v2['is_leap_year'] = is_leap_year.apply(lambda x: 1 if x == True else 0) 

# Monthly statistics
ss_ddf_v2['month'] = ss_ddf_v2.time_key.apply(lambda x: x.month)
ss_ddf_v2['week_of_month'] = ss_ddf_v2.time_key.apply(lambda x: 1 if x.day < 8 else (2 if x.day > 7 and x.day < 15 else (3 if x.day > 14 and x.day < 24 else 4)))
ss_ddf_v2['first_month_half'] = ss_ddf_v2.week_of_month.apply(lambda x: 1 if x == 1 or x == 2 else 0) 
ss_ddf_v2['second_month_half'] = ss_ddf_v2.week_of_month.apply(lambda x: 1 if x == 3 or x == 4 else 0) 
ss_ddf_v2['is_month_start'] = ss_ddf_v2.week_of_month.apply(lambda x: 1 if x == 1 else 0)
ss_ddf_v2['is_month_end'] = ss_ddf_v2.week_of_month.apply(lambda x: 1 if x == 4 else 0) 
ss_ddf_v2['month_days'] = ss_ddf_v2.time_key.apply(lambda x: x.days_in_month) 

# Left padded strings shopping star/end hour with 0's
# Some strings (morning before 10 am and after midnight) have a length < 6
shop_start_hour = ss_ddf_v2['shop_trip_start_hour_key'].apply(lambda x: (6 - len(x)) * '0' + x)
shop_end_hour = ss_ddf_v2['shop_trip_end_hour_key'].apply(lambda x: (6 - len(x)) * '0' + x)

# Concatenating time_key and shopping start/end hour attributes and converting to datetime
ss_ddf_v2['shop_trip_start_hour_key'] = (ss_ddf_v2.time_key.astype(str) + ' ' + shop_start_hour).astype('datetime64')
# TODO - Add extra day to time_key (+ shop_end) if snop_start before midnight and shop_end after midnight
ss_ddf_v2['shop_trip_end_hour_key'] = (ss_ddf_v2.time_key.astype(str) + ' ' + shop_end_hour).astype('datetime64')

# Time of day the shopping started
ss_ddf_v2['shop_started_morning'] = ss_ddf_v2.shop_trip_start_hour_key.apply(lambda x: 1 if x.hour >= 7 and x.hour < 13 else 0)
ss_ddf_v2['shop_started_evening'] = ss_ddf_v2.shop_trip_start_hour_key.apply(lambda x: 1 if x.hour >= 13 and x.hour < 20 else 0)
ss_ddf_v2['shop_started_night'] = ss_ddf_v2.shop_trip_start_hour_key.apply(lambda x: 1 if x.hour >= 20 or x.hour < 7 else 0)

# Time of day the shopping ended
ss_ddf_v2['shop_ended_morning'] = ss_ddf_v2.shop_trip_end_hour_key.apply(lambda x: 1 if x.hour >= 7 and x.hour < 13 else 0)
ss_ddf_v2['shop_ended_evening'] = ss_ddf_v2.shop_trip_end_hour_key.apply(lambda x: 1 if x.hour >= 13 and x.hour < 20 else 0)
ss_ddf_v2['shop_ended_night'] = ss_ddf_v2.shop_trip_end_hour_key.apply(lambda x: 1 if x.hour >= 20 or x.hour < 7 else 0)

# Elapsed time (in days) since the client registered in the ss
ss_ddf_v2['elapsed_register_time'] = (ss_ddf_v2.time_key - ss_ddf_v2.ss_register_time_key).dt.days

# Total shopping time (in seconds)
ss_ddf_v2['total_shopping_time'] = ((ss_ddf_v2['shop_trip_end_hour_key'] - ss_ddf_v2['shop_trip_start_hour_key']).dt.total_seconds()).astype(int)

# Tempo medio, em segundos, por produto comprado
ss_ddf_v2['mean_time_per_product'] = ss_ddf_v2.transactional_total_qt / ss_ddf_v2.total_shopping_time

# Removing attributes useless for the prediction
ss_ddf_v2 = ss_ddf_v2.drop(['time_key', 'card_key', 'ss_register_time_key', 'shop_trip_start_hour_key', 'shop_trip_end_hour_key'], axis=1)

'''
### DATASET V2 ###
For this data set, data types have been adjusted (some quantities converted to ints)
and more time related attributes have been added. Also, non-predictive attributes have been removed (time_key, card_key, etc).
- Time attributes (yearly, monthly, daily, etc)
- Time of day
- Total shopping time
- Mean time per product
'''

#ss_ddf_v2.to_csv('/home/routar/FEUP/workspace/SONAE/ss_ml/newdata/remod/ss_filt_data_v2-*.csv', sep=';', header=False, index=False)

# Keeping v2 data set unchanged
ss_ddf_v3 = ss_ddf_v2

# Converting quantities to integer values
ss_ddf_v3.product_added_times_qt = ss_ddf_v3.product_added_times_qt.astype(int)
ss_ddf_v3.product_removed_qt = ss_ddf_v3.product_removed_qt.astype(int)

# Avg price per product
ss_ddf_v3['avg_price_product'] = ss_ddf_v3.transactional_total_amt_eur / ss_ddf_v3.transactional_total_qt

# Difference between total scanned added products and total amount products
ss_ddf_v3['diff_total_added_products'] = ss_ddf_v3.product_added_times_qt - ss_ddf_v3.transactional_total_qt

# Difference between total scanned removed products and total amount products
ss_ddf_v3['diff_total_removed_products'] = ss_ddf_v3.product_removed_qt - ss_ddf_v3.transactional_total_qt

# Ratio between total scanned added products and total amount products
ss_ddf_v3['ratio_total_added_products'] = ss_ddf_v3.product_added_times_qt / ss_ddf_v3.transactional_total_qt

# Ratio between total scanned removed products and total amount products
ss_ddf_v3['ratio_total_removed_products'] = ss_ddf_v3.product_removed_qt / ss_ddf_v3.transactional_total_qt

#{ echo "transactional_total_amt_eur;transactional_total_qt;total_divergence_fl;unknown_product_fl;product_added_times_qt;product_removed_qt;number_different_products;day;week_day;is_weekend;year;year_day;week_of_year;is_quarter_end;is_quarter_start;is_year_end;is_year_start;is_leap_year;month;week_of_month;first_month_half;second_month_half;is_month_start;is_month_end;month_days;shop_started_morning;shop_started_evening;shop_started_night;shop_ended_morning;shop_ended_evening;shop_ended_night;elapsed_register_time;total_shopping_time;mean_time_per_product;avg_price_product;diff_total_added_products;diff_total_removed_products;ratio_total_added_products;ratio_total_removed_products"; cat ss_v3.csv; } > ss_v3.1.csv

'''
### DATASET V3 ###
For this data set, new attributes have been added.
- Average price per product
- Product scanned added/removed times 
- Difference/Ratio between total scanned products and total products in transaction
- Difference/Ratio between total removed scanned products and total products in transaction
- Number of different products
'''

#ss_ddf_v3.to_csv('/home/routar/FEUP/workspace/SONAE/ss_ml/newdata/remod/ss_filt_data_v3-*.csv', sep=';', header=False, index=False)

# Keeping v3 data set unchanged
ss_ddf_v4 = ss_ddf_v3

'''
### DATASET V4 ###
For this data set, the transactional divergence score has been added
'''

'''
### V4 PERFORMANCE ###

# Gradient Boosted Tree #
- All attributes
0 class recall: 95.48%
0 class precision: 94.72%
1 class recall: 24.15% 
1 class precision: 27.26%

- Top 12 weight attributes
0 class recall: 95.20%
0 class precision: 94.78%
1 class recall: 25.21% 
1 class precision: 26.91%

# Deep Learning #
0 class recall: 96.20%
0 class precision: 94.64%
1 class recall: 22.25% 
1 class precision: 29.10%
'''

#ss_ddf_v4.to_csv('/home/routar/FEUP/workspace/SONAE/ss_ml/newdata/remod/ss_filt_data_v4-*.csv', sep=';', header=False, index=False)

#{ echo "transactional_total_amt_eur;transactional_total_qt;total_divergence_fl;unknown_product_fl;product_added_times_qt;product_removed_qt;number_different_products;transactional_div_score;day;week_day;is_weekend;year;year_day;week_of_year;is_quarter_end;is_quarter_start;is_year_end;is_year_start;is_leap_year;month;week_of_month;first_month_half;second_month_half;is_month_start;is_month_end;month_days;shop_started_morning;shop_started_evening;shop_started_night;shop_ended_morning;shop_ended_evening;shop_ended_night;elapsed_register_time;total_shopping_time;mean_time_per_product;avg_price_product;diff_total_added_products;diff_total_removed_products;ratio_total_added_products;ratio_total_removed_products"; cat ss_v4.csv; } > ss_v4.1.csv

ss_ddf_v5 = ss_ddf_v4

# List of optional holidays (names defined by holidays 0.9.7)
optional_holid = [
    'Carnaval',
    'Vespera de Natal',
    '26 de Dezembro',
    'Vespera de Ano novo'
]

# List of religious holidays (names defined by holidays 0.9.7)
relig_holid = [
    'Sexta-feira Santa', 
    'Páscoa', 
    'Corpo de Deus', 
    'Assunção de Nossa Senhora', 
    'Dia de Todos os Santos', 
    'Imaculada Conceição',
    'Vespera de Natal',
    'Christmas Day'
]

# Mapping holidays dates, names and types to a dictionay
# e.g. date(2018,12,25): ('Christmas Day', ['religious, national])
def map_holidays(pt_holidays):
    holidays_dic = dict()
    for hdate, name in pt_holidays.items():
        if name in optional_holid:
            if name in relig_holid:
                holidays_dic[hdate] = (name, ['religious', 'optional'])
            else:
                holidays_dic[hdate] = (name, ['optional'])
        else:
            if name in relig_holid:
                holidays_dic[hdate] = (name, ['religious', 'national'])
            else:
                holidays_dic[hdate] = (name, ['national'])
    return holidays_dic

def holidays_fn(x):
    # Getting dates for Portuguese holidays
    pt_holidays = holidays.PortugalExt(years=int(x.year))
    shopping_date = date(int(x.year),int(x.month),int(x.day))
    holidays_dic = map_holidays(pt_holidays)
    if shopping_date in holidays_dic:
        return holidays_dic[shopping_date][1]
    return ['null']

# 1 hot encoding types of holidays
holiday_type = ss_ddf_v5.apply(holidays_fn, axis=1)
ss_ddf_v5['religious_holiday'] = holiday_type.apply(lambda x: 1 if 'religious' in x else 0)
ss_ddf_v5['optional_holiday'] = holiday_type.apply(lambda x: 1 if 'optional' in x else 0)
ss_ddf_v5['national_holiday'] = holiday_type.apply(lambda x: 1 if 'national' in x else 0)

def set_season(x):
    # Season
    spring = range(80, 172)
    summer = range(172, 264)
    fall = range(264, 355)
    if x in spring:
        return 'spring'
    elif x in summer:
        return 'summer'
    elif x in fall:
        return 'fall'
    else:
        return 'winter'

season = ss_ddf_v5.year_day.apply(set_season)
ss_ddf_v5['is_spring'] = season.apply(lambda x: 1 if x == 'spring' else 0)
ss_ddf_v5['is_summer'] = season.apply(lambda x: 1 if x == 'summer' else 0)
ss_ddf_v5['is_fall'] = season.apply(lambda x: 1 if x == 'fall' else 0)
ss_ddf_v5['is_winter'] = season.apply(lambda x: 1 if x == 'winter' else 0)

'''
### DATASET V5 ###
For this data set, new attributes have been added:
- holidays booleans (religious, optional, national)
- yearly season
'''

#ss_ddf_v5.to_csv('/home/routar/FEUP/workspace/SONAE/ss_ml/newdata/remod/ss_filt_data_v5-*.csv', sep=';', header=False, index=False)
#{ echo "transactional_total_amt_eur;transactional_total_qt;total_divergence_fl;unknown_product_fl;product_added_times_qt;product_removed_qt;number_different_products;transactional_div_score;day;week_day;is_weekend;year;year_day;week_of_year;is_quarter_end;is_quarter_start;is_year_end;is_year_start;is_leap_year;month;week_of_month;first_month_half;second_month_half;is_month_start;is_month_end;month_days;shop_started_morning;shop_started_evening;shop_started_night;shop_ended_morning;shop_ended_evening;shop_ended_night;elapsed_register_time;total_shopping_time;mean_time_per_product;avg_price_product;diff_total_added_products;diff_total_removed_products;ratio_total_added_products;ratio_total_removed_products;religious_holiday;optional_holiday;national_holiday;is_spring;is_summer;is_autumn;is_winter"; cat ss_v5.csv; } > ss_v5.1.csv

#prod_ddf

######################
# DATA NORMALIZATION #
######################

# minmax scaler 
'''
ss_ddf_v5 = ss_ddf_v5.compute()
scaler = MinMaxScaler()
ss_ddf_v5[ss_ddf_v5.columns] = scaler.fit_transform(ss_ddf_v5[ss_ddf_v5.columns])

ss_ddf_v5.to_csv('/home/routar/FEUP/workspace/SONAE/ss_ml/data/processed/gen/ss_filt_data_v5-*.csv', sep=';', header=False, index=False)
'''
#{ echo "transactional_total_amt_eur;transactional_total_qt;total_divergence_fl;unknown_product_fl;product_added_times_qt;product_removed_qt;number_different_products;transactional_div_score;day;week_day;is_weekend;year;year_day;week_of_year;is_quarter_end;is_quarter_start;is_year_end;is_year_start;is_leap_year;month;week_of_month;first_month_half;second_month_half;is_month_start;is_month_end;month_days;shop_started_morning;shop_started_evening;shop_started_night;shop_ended_morning;shop_ended_evening;shop_ended_night;elapsed_register_time;total_shopping_time;mean_time_per_product;avg_price_product;diff_total_added_products;diff_total_removed_products;ratio_total_added_products;ratio_total_removed_products;religious_holiday;optional_holiday;national_holiday;is_spring;is_summer;is_autumn;is_winter"; cat ss_v5.csv; } > ss_v5.1.csv

###################
# DATA CONVERSION #
###################

# Escrita to dataframe para csv (em diferentes particoes)
#ss_ddf.to_csv('/home/routar/FEUP/workspace/SONAE/ss_ml/newdata/ss_filt_data-*.csv', sep=';', header=False, index=False)
#ss_ddf.to_csv('/home/routar/FEUP/workspace/SONAE/ss_ml/newdata/remod/ss_filt_data-*.csv', sep=';', header=False, index=False)
# Comando para fazer merge das particoes csv criadas
#sed 1d ss_filt_data-*.csv > merged.csv

#echo 'transactional_total_amt_eur;transactional_total_qt;total_divergence_fl;product_added_times_qt;product_removed_qt;number_different_products;day;week_day;is_weekend;year;year_day;week_of_year;is_quarter_end;is_quarter_start;is_year_end;is_year_start;is_leap_year;month;week_of_month;first_month_half;second_month_half;is_month_start;is_month_end;month_days;elapsed_register_time;total_shopping_time;shop_started_morning;shop_started_evening;shop_started_night;shop_ended_morning;shop_ended_evening;shop_ended_night;mean_time_per_product;avg_price_product;ratio_total_added_products;ratio_total_removed_products'
#cat in.csv >> out.csv
#{ echo "transactional_total_amt_eur;transactional_total_qt;total_divergence_fl;product_added_times_qt;product_removed_qt;number_different_products;day;week_day;is_weekend;year;year_day;week_of_year;is_quarter_end;is_quarter_start;is_year_end;is_year_start;is_leap_year;month;week_of_month;first_month_half;second_month_half;is_month_start;is_month_end;month_days;elapsed_register_time;total_shopping_time;shop_started_morning;shop_started_evening;shop_started_night;shop_ended_morning;shop_ended_evening;shop_ended_night;mean_time_per_product"; cat ss_v1.csv; } > ss_v2.csv

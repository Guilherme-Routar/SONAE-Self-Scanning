import dask.dataframe as dd
import pandas as pd

# Defining attributes' names
header = [
    'time_key',                     # Date of the transaction (yyyymmdd)
    'location_key',                 # Shop location id
    'transaction_id',               # Id of the transaction
    'system_key',                   # System's id
    'card_key',                     # Clien
    'scanner_id',                   # Scanner's id
    'ss_register_time_key',         # Time of first register of the client (yyyymmdd)
    'shop_trip_start_hour_key',     # Start hour of the shop trip (hmmss)
    'shop_trip_end_hour_key',       # End hour of the shop trip (hmmss)
    'transaction_end_hour_key',     # End hour of the transaction (payment) (hmmss)
    'rescan_fl',                    # 0: transaction not rescanned, 1: transaction rescanned
    'rescan_tp',                    # Type of the rescan: NONE, PARTIAL, COMPLETE
    'rescan_start_hour_key',        # Rescan start time of transaction (hmmss)
    'rescan_end_hour_key',          # Rescan end hour of product or transaction
    'transactional_total_amt',      # Total amount of the transaction in country's currency (all in euros)
    'transactional_total_amt_eur',  # Total amount of the transaction in euros
    'transactional_total_qt',       # Total amount of products in transaction
    'total_divergence_fl',          # 0: no divergence, 1: divergence
    'total_divergence_sign',        # =: same amount, -: scanned extra items +: scanned less items
    'total_divergence_amt',         # Amount of divergence in country's currency (all in euros)
    'total_divergence_amt_eur',     # Amount of divergence in euros
    'total_divergence_qt',          # Amount of divergent products
    'ean_key',                      # id of product (shared by same products) 
    'product_added_times_qt',       # Amount of times the product was scanned by the customer
    'product_removed_qt',           # Amount of times the product was removed by the customer
    'unknown_product_fl',           # 1: scanner failed to identify product, 0: scanner didn't fail 
    'rescan_product_fl',            # 0: product wasn't rescanned, 1: product was rescanned
    'product_total_amt',            # Total amount of the product country's currency (all in euros) (actually scanned by the customer)
    'product_total_amt_eur',        # Total amount of the product in euros (actually scanned by the customer)
    'product_total_qt',             # Total amount of the product (actually scanned by the customer)
    'product_divergence_fl',        # 0: product was correctly scanned, 1: product wasn't correctly scanned
    'product_divergence_sign',      # =: same amount, -: overly scanned, +: underly scanned
    'product_divergence_amt',       # Total amount of the divergence in country's currency (all in euros)
    'product_divergence_amt_eur',   # Total amount of the divergence in euros
    'product_divergence_qt',        # Amount of products in divergence
    'fl_abandoned',                 # Client left while being audited (all 0)
    'currency_key',                 # Id of the currency (all euros)
    'create_date',                  
    'last_updt_date',
    'loc_brand_cd',                 # IGNORE
    'chain_cd',                     # IGNORE
    'department_cd',                # IGNORE
    'club_cd',                      # IGNORE
    'month_key'
    ]

# Atributos sem importância para a previsão de divergência
drop_atts = [
    'scanner_id',
    'system_key',
    'transactional_total_amt',
    'total_divergence_amt',
    'product_total_amt',
    'product_divergence_sign',
    'product_divergence_amt',
    'fl_abandoned',
    'currency_key',
    'create_date',
    'last_updt_date',
    'loc_brand_cd',
    'chain_cd',
    'department_cd',
    'club_cd',
    'month_key'
]

types = {
    'ean_key': 'object',
    'transaction_end_hour_key': 'object'
}

#aggr_key=1
#key_status=1

# O data set inicial (sem alterações) tem, aproximadamente, 40 milhoes de linhas/instancias
ss_ddf = dd.read_csv(
    'data/original/original_ss.csv',
    sep=';',
    names=header,
    dtype = types,
).drop(drop_atts, axis=1)

# Filtro para ficar apenas com as transações auditadas (aprendizagem supervisionada)
ss_ddf = ss_ddf.query("rescan_fl == True")

# Filtro para ficar apenas com as lojas mais ativas a usar SS
#active_shops = [1, 2, 3, 7, 10, 12, 463, 1393]
#ss_ddf = ss_ddf[(ss_ddf.location_key // 1000000).isin(active_shops)]

# Conversão do dataframe para csv
#ss_ddf.to_csv('/home/routar/FEUP/workspace/SONAE/ss_ml/newdata/filter-*.csv', sep=';', index=False, header=False)
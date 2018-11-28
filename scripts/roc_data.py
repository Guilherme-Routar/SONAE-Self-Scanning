import dask.dataframe as dd
import matplotlib.pyplot as plt

baseline_ddf = dd.read_csv(
    'data/processed/confs/baseline_conf.csv',
    sep = ';'
).drop('confidence(0)', axis=1)

final_ddf = dd.read_csv(
    'data/processed/confs/ss_v5.1_allshops_conf.csv',
    sep = ';'
).drop('confidence(0.0)', axis=1)

def compute_ROC(ddf):
    # Calculating whether it was a correct or wrong prediction
    ddf['wrong'] = abs(ddf['prediction(total_divergence_fl)'] - ddf['total_divergence_fl']).astype(int)

    # Grouping by the type of the prediction
    ddf_count = ddf.groupby('total_divergence_fl').agg('count')
    ddf_count = ddf_count.compute()
    ddf_count.index = ddf_count.index.map(int)

    # Getting number of wrong and correct predictions
    wrong_prediction_count = ddf_count.at[1, 'wrong']
    correct_prediction_count = ddf_count.at[0, 'wrong']

    return compute_TFPR(ddf.compute(), wrong_prediction_count, correct_prediction_count)

def compute_TFPR(ddf, wrong_prediction_count, correct_prediction_count):
    n = ddf.loc[0, 'wrong']
    if n == 1:
        ddf.loc[0, 'TPR'] = 0
        ddf.loc[0, 'FPR'] = 1 / wrong_prediction_count
    else:
        ddf.loc[0, 'TPR'] = 1 / correct_prediction_count
        ddf.loc[0, 'FPR'] = 0
    
    #print(wrong_prediction_count)
    #print(correct_prediction_count)

    for i in range(1, len(ddf)):
        if ddf.loc[i, 'wrong'] == 1:
            ddf.loc[i, 'TPR'] = ddf.loc[i - 1, 'TPR']
            ddf.loc[i, 'FPR'] = ddf.loc[i - 1, 'FPR'] + (1 / wrong_prediction_count)
        else:
            ddf.loc[i, 'TPR'] = ddf.loc[i - 1, 'TPR'] + (1 / correct_prediction_count)
            ddf.loc[i, 'FPR'] = ddf.loc[i - 1, 'FPR']
    return ddf

baseline_ddf = compute_ROC(baseline_ddf)
final_ddf = compute_ROC(final_ddf)

ax = baseline_ddf.plot.scatter(x='FPR', y='TPR')
final_ddf.plot.scatter(ax=ax, x='FPR', y='TPR')

plt.show()

#ddf.to_csv('/home/routar/FEUP/workspace/SONAE/ss_ml/data/roc.csv', columns=['FPR', 'TPR'], sep=';', header=False, index=False)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
re_matrix=pd.read_pickle('plk/return_matrix.plk')
mean_series=re_matrix.mean()
mean_series.to_pickle('plk/mean_series.plk')
corr_matrix=re_matrix.corr()
corr_matrix.to_pickle('plk/corr_matrix.plk')

def plot_corr(corr,size=10):
    fig,ax=plt.subplots(figsize=(size,size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)),corr.columns);
    plt.yticks(range(len(corr.columns)),corr.columns);

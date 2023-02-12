import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def loss_plot(exp_dir_path):
    results_df = pd.read_csv(os.path.join(exp_dir_path, 'loss_results.csv'), header=0, index_col=0)
    agg_results = results_df.mean(axis=0).to_list()
    xticks = [x + 1 for x in range(len(agg_results))]
    ax = plt.figure().gca()
    ax.plot(xticks, agg_results, 'b', label='Training loss')
    plt.title('Training loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(os.path.join(exp_dir_path, 'loss_plt.png'))
    plt.close()
    print('Loss plot saved to {}'.format(os.path.join(exp_dir_path, 'loss_plt.png')))
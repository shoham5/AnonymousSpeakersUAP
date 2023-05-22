import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import time

import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import sys
import matplotlib.pyplot as plt
# from utils.data_utils import get_test_loaders # circular
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import seaborn as sns
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  time_axis = torch.arange(0, num_frames) / sample_rate

  figure, axes = plt.subplots(num_channels, 1)
  if num_channels == 1:
    axes = [axes]
  for c in range(num_channels):
    axes[c].plot(time_axis, waveform[c], linewidth=1)
    axes[c].grid(True)
    if num_channels > 1:
      axes[c].set_ylabel(f'Channel {c+1}')
    if xlim:
      axes[c].set_xlim(xlim)
    if ylim:
      axes[c].set_ylim(ylim)
  figure.suptitle(title)
  plt.ylim([-0.7, 0.7])
  plt.show(block=False)

def plot_train_val_loss(config, loss, loss_type):
    xticks = [x + 1 for x in range(len(loss))]
    plt.plot(xticks, loss, 'b', label='Training loss')
    plt.title('Training loss')
    plt.xlabel(loss_type)
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.savefig(config.current_dir + '/final_results/train_loss_' + loss_type.lower() + '_plt.png')
    plt.close()


def plot_separate_loss(config, train_losses_epoch, dist_losses, tv_losses):
    epochs = [x + 1 for x in range(len(train_losses_epoch))]
    weights = np.array([config.dist_weight, config.tv_weight])
    number_of_subplots = weights[weights > 0].astype(bool).sum()
    fig, axes = plt.subplots(nrows=1, ncols=number_of_subplots, figsize=(6 * number_of_subplots, 2 * number_of_subplots), squeeze=False)
    idx = 0
    for weight, train_loss, label in zip(weights, [dist_losses,  tv_losses], ['Distance loss', 'Total Variation loss']):
        if weight > 0:
            axes[0, idx].plot(epochs, train_loss, c='b', label='Train')
            axes[0, idx].set_xlabel('Epoch')
            axes[0, idx].set_ylabel('Loss')
            axes[0, idx].set_title(label)
            axes[0, idx].legend(loc='upper right')
            axes[0, idx].xaxis.set_major_locator(MaxNLocator(integer=True))
            idx += 1
    fig.tight_layout()
    plt.savefig(config.current_dir + '/final_results/separate_loss_plt.png')
    plt.close()

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


def sims_plot(exp_dir_path):
    results_df = pd.read_csv(os.path.join(exp_dir_path, 'sims_results.csv'), header=0, index_col=0)
    agg_results = results_df.mean(axis=0).to_list()
    xticks = [x + 1 for x in range(len(agg_results))]
    ax = plt.figure().gca()
    ax.plot(xticks, agg_results, 'b', label='Evaluation Similarities Plot')
    plt.title('Evaluation Similarities')
    plt.xlabel('Iteration')
    plt.ylabel('Similarity')
    plt.legend(loc='upper right')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(os.path.join(exp_dir_path, 'sims_plt.png'))
    plt.close()
    print('Sims plot saved to {}'.format(os.path.join(exp_dir_path, 'sims_plt.png')))

def snr_plot(exp_dir_path, eval_snr='Evaluate snr'):
    results_df = pd.read_csv(os.path.join(exp_dir_path, f'{eval_snr}_results.csv'), header=0, index_col=0)
    agg_results = results_df.mean(axis=0).to_list()
    xticks = [x + 1 for x in range(len(agg_results))]
    ax = plt.figure().gca()
    ax.plot(xticks, agg_results, 'b', label=f'{eval_snr}')
    plt.title(f'{eval_snr}')
    plt.xlabel('Iteration')
    plt.ylabel('SNR')
    plt.legend(loc='upper right')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(os.path.join(exp_dir_path, 'snr_plt.png'))
    plt.close()
    print('SNR plot saved to {}'.format(os.path.join(exp_dir_path, f'{eval_snr}_plt.png')))


class Plots:
    def __init__(self, root_path, x, y):
        self.X = x
        self.y = y
        self.features_cols = init_features_cols(x)
        self.df = convert_to_df(x, y,self.features_cols)
        self.random_permutation = self.init_random_permutation()


    def init_random_permutation(self):
        np.random.seed(42)
        return np.random.permutation(self.df.shape[0])

    def reduction_dims_using_pca(self):
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(self.df[self.features_cols].values)

        self.df['pca-one'] = pca_result[:, 0]
        self.df['pca-two'] = pca_result[:, 1]
        self.df['pca-three'] = pca_result[:, 2]

        print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

    def generate_samples_plot_mnist(self):
        plt.gray()  # for images
        fig = plt.figure(figsize=(16, 7))
        for i in range(0, 15):
            ax = fig.add_subplot(3, 5, i + 1, title="Digit: {}".format(str(self.df.loc[self.random_permutation[i], 'label'])))
            ax.matshow(self.df.loc[self.random_permutation[i], self.features_cols].values.reshape((28, 28)).astype(float))
        plt.show()

def init_features_cols(X):
    return ['pixel' + str(i) for i in range(X.shape[1])]

def plot_using_pca_2d(df,y, rndperm ):
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x=df["pca-one"], y=df['pca-two'],
        hue=df['y'],
        palette=sns.color_palette("hls", 10),
        data=self.df.loc[self.random_permutation,:],
        legend="full",
        alpha=0.3
    )

def convert_to_df(X, y,cols):

    df = pd.DataFrame(X, columns=cols)
    df['y'] = y
    df['label'] = df['y'].apply(lambda i: str(i))

    X, y = None, None

    print('Size of the dataframe: {}'.format(df.shape))
    # df.fillna(0)
    # randomize sampels
    return df.fillna(0)
    # generate_samples_plot(df, feat_cols, rndperm)  # to plot
    # reduction_dims_using_pca()  # to create pca




def plot_using_pca_3d():
    ax = plt.figure(figsize=(16, 10)).gca(projection='3d')
    ax.scatter(
        xs=df.loc[rndperm, :]["pca-one"],
        ys=df.loc[rndperm, :]["pca-two"],
        zs=df.loc[rndperm, :]["pca-three"],
        c=df.loc[rndperm, :]["y"],
        cmap='tab10'
    )
    ax.set_xlabel('pca-one')
    ax.set_ylabel('pca-two')
    ax.set_zlabel('pca-three')
    plt.show()


def plot_tsne(exp_dir_path):
    N = 10000

    df_subset = df.loc[rndperm[:N], :].copy()

    data_subset = df_subset[feat_cols].values

    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(data_subset)

    df_subset['pca-one'] = pca_result[:, 0]
    df_subset['pca-two'] = pca_result[:, 1]
    df_subset['pca-three'] = pca_result[:, 2]

    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(data_subset)

    print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))
    df_subset['tsne-2d-one'] = tsne_results[:, 0]
    df_subset['tsne-2d-two'] = tsne_results[:, 1]

    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("hls", 10),
        data=df_subset,
        legend="full",
        alpha=0.3
    )


def compare_pca_tsne():
    plt.figure(figsize=(16, 7))

    ax1 = plt.subplot(1, 2, 1)
    sns.scatterplot(
        x="pca-one", y="pca-two",
        hue="y",
        palette=sns.color_palette("hls", 10),
        data=df_subset,
        legend="full",
        alpha=0.3,
        ax=ax1
    )

    ax2 = plt.subplot(1, 2, 2)
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("hls", 10),
        data=df_subset,
        legend="full",
        alpha=0.3,
        ax=ax2
    )


def pca_then_tsne():
    pca_50 = PCA(n_components=50)
    pca_result_50 = pca_50.fit_transform(data_subset)

    print('Cumulative explained variation for 50 principal components: {}'.format(
        np.sum(pca_50.explained_variance_ratio_)))

    time_start = time.time()

    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
    tsne_pca_results = tsne.fit_transform(pca_result_50)

    print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

    df_subset['tsne-pca50-one'] = tsne_pca_results[:, 0]
    df_subset['tsne-pca50-two'] = tsne_pca_results[:, 1]
    plt.figure(figsize=(16, 4))
    ax1 = plt.subplot(1, 3, 1)
    sns.scatterplot(
        x="pca-one", y="pca-two",
        hue="y",
        palette=sns.color_palette("hls", 10),
        data=df_subset,
        legend="full",
        alpha=0.3,
        ax=ax1
    )

    ax2 = plt.subplot(1, 3, 2)
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("hls", 10),
        data=df_subset,
        legend="full",
        alpha=0.3,
        ax=ax2
    )

    ax3 = plt.subplot(1, 3, 3)
    sns.scatterplot(
        x="tsne-pca50-one", y="tsne-pca50-two",
        hue="y",
        palette=sns.color_palette("hls", 10),
        data=df_subset,
        legend="full",
        alpha=0.3,
        ax=ax3
    )

def set_dataset(self, dataset_name):
    with open(ROOT / 'configs/dataset_config.yaml', 'r') as stream:
        return yaml.safe_load(stream)[dataset_name]

def get_data_from_speakers_dataset():
    self.test_img_dir = {name: self.set_dataset(name) for name in self.test_dataset_names}
    self.test_number_of_people = 20
    self.test_celeb_lab = {}

    for dataset_name, img_dir in self.test_img_dir.items():
        label_list = os.listdir(img_dir['root_path'])[:self.test_number_of_people]
        if dataset_name == self.train_dataset_name:
            label_list = os.listdir(img_dir['root_path'])[-self.test_number_of_people:]
        self.test_celeb_lab[dataset_name] = label_list
    self.test_celeb_lab_mapper = {dataset_name: {i: lab for i, lab in enumerate(self.test_celeb_lab[dataset_name])}
                                  for dataset_name in self.test_dataset_names}
    emb_loaders, test_loaders = get_test_loaders(self.config,self.config.test_celeb_lab.keys())


def sort_by_target(mnist):
    reorder_train = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[:60000])]))[:, 1]
    reorder_test = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[60000:])]))[:, 1]
    # mnist.data[:60000] = np.array(mnist.data)[reorder_train]
    # mnist.data[:60000] = mnist.data.iloc[reorder_train]
    mnist.target[:60000] = mnist.target[reorder_train]
    mnist.data[60000:] = mnist.data.iloc[(reorder_test + 60000)]
    mnist.target[60000:] = mnist.target[(reorder_test + 60000)]

def get_mnist_data():

    try:
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml('mnist_784', version=1, cache=True)
        mnist.target = mnist.target.astype(np.int8)  # fetch_openml() returns targets as strings
        sort_by_target(mnist)  # fetch_openml() returns an unsorted dataset
    except ImportError:
        from sklearn.datasets import fetch_mldata
        mnist = fetch_mldata('MNIST original')

    X = mnist.data / 255.0
    y = mnist.target

    print(X.shape, y.shape)
    return X, y
    # convert_to_df(X, y)

def main():
    X , y = get_mnist_data()
    root_path = ""
    plot_obj = Plots(root_path=root_path, x=X, y = y)
    plot_obj.generate_samples_plot_mnist()
    plot_obj.reduction_dims_using_pca()
    print("finish")



if __name__=='__main__':
    main()

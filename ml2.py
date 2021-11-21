# t-SNE, UMAP, Principal Component Analysis (PCA)

import pandas as pd
import dataget
import gensim.downloader as api
import numpy as np
import seaborn as sns
from scipy import io
from matplotlib import pyplot as plt

import umap
# import umap.plot
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits


# digits = load_digits()
# embedding = umap.UMAP(n_neighbors=5,
#                       min_dist=0.3,
#                       metric='correlation').fit_transform(digits.data)


mnist = load_digits()
# print (mnist.data.shape, 'F_mnist SHAPE')

_, __, f_mnist, f_mnist_target = dataget.image.fashion_mnist().get()
f_mnist = f_mnist.reshape(-1, 28*28)
# print (f_mnist.shape, 'mnist SHAPE')


mat = io.loadmat('COIL20.mat')
coil20 = mat['X']
# print (coil20.shape, 'coil20 SHAPE')

tsne = TSNE(n_jobs=-1)
pca = PCA(n_components=2)
umap = umap.UMAP()
algs = [umap, tsne, pca] #tsne, pca


def plotter (fig, title, embeddings, targ, ax):
    x = embeddings[:, 0]
    y = embeddings[:, 1]
    if targ.size != 0:
        scatter = ax.scatter(x, y, c=targ, cmap = 'Spectral', marker = '.')
    else:
        scatter = ax.scatter(x, y, marker='.')
    ax.set_title(title)

    fig.colorbar(scatter, ax=ax)


def FMNIST_MNIST_COIL20():
    plt.rcParams["figure.figsize"] = (13, 13)

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, constrained_layout=True)

    def plotter_init(dataset, target, axes_array):
        data, target, axes = dataset, target, axes_array
        for alg, ax in np.vstack([algs, axes]).T:
            x_trans = alg.fit_transform(data)
            plotter(fig, type(alg).__name__, x_trans, target, ax)

    plotter_init(mnist.data, mnist.target, [ax1, ax4, ax7]) #mnist
    plotter_init(f_mnist, f_mnist_target, [ax2, ax5, ax8]) #f_mnist
    plotter_init(coil20, mat['Y'], [ax3, ax6, ax9]) #coil20

    fig.suptitle(' MNIST ::::: //// ::::: FASHION MNIST ::::: //// ::::: COIL20 ', fontsize=9)
    plt.show()


def GOOGLE_NEWS():
    plt.rcParams["figure.figsize"] = (10, 7)

    google_news = api.load("word2vec-google-news-300").vectors
    google_news = google_news[np.random.randint(google_news.shape[0],
                    size=int(google_news.shape[0]/50)), :]
    # print (googl_news.shape, 'SHAPE???')

    #UMAP ~1min
    #TSNE ~30min
    #PCA <1min
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, constrained_layout=True)

    def plotter_init_google(axes_array):
        axes = axes_array
        for alg, ax in np.vstack([algs, axes]).T:
            x_trans = alg.fit_transform(google_news)
            plotter(fig, type(alg).__name__, x_trans, np.array([]), ax)

    plotter_init_google([ax1, ax2, ax3])
    fig.suptitle(' /// GOOGLE NEWS ///')
    plt.show()


#Выберите одну из функций для данных FASHION MNIST, MNIST, COIL20 или GOOGLE NEWS
FMNIST_MNIST_COIL20()
# GOOGLE_NEWS()

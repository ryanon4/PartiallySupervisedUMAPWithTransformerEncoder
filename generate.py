import joblib
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.datasets import fetch_20newsgroups
import sklearn
import umap
import tensorflow as tf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from encoder import build_model, attention
from sklearn.manifold import TSNE
def generate_vectors(algorithm, target, vectors, dimension,limit=-1, test_size=0.8, batch_size=64):
    # Load Original BERT Vectors
    # Load Dataset

    target = target#[0:limit]
    vectors = vectors#[0:limit]

    dims = vectors[0].shape

    #Non-Supervised Algorithms
    if algorithm in ["PCA", "UMAP", "PARAMETRIC_UMAP", "TSNE"]:
        if algorithm == "PCA":
            reduced_vectors = sklearn.decomposition.PCA(n_components=dimension).fit_transform(vectors)

        elif algorithm == "UMAP":
            umap_args = {'n_neighbors': 15,
                         'n_components': dimension,
                         'metric': 'cosine'}
            reduced_vectors = umap.UMAP(**umap_args).fit(vectors).embedding_

        elif algorithm == "PARAMETRIC_UMAP":
            umap_args = {'n_neighbors': 15,
                         'n_components': dimension,
                         'metric': 'cosine',
                         "keras_fit_kwargs":{"verbose":1}}
            reducer = umap.parametric_umap.ParametricUMAP(**umap_args).fit(vectors)
            reduced_vectors = reducer.transform(vectors)
        elif algorithm == "TSNE":
            model = TSNE(n_components=dimension, n_jobs=-1, method="exact")
            reduced_vectors = model.fit_transform(vectors)

    #Supervised Algorithms
    elif algorithm in ["LDA", "UMAP_SUPERVISED", "PARAMETRIC_UMAP_SUPERVISED", "UMAP_SUPERVISED_RNN", "UMAP_SUPERVISED_TRANSFORMER"]:
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            vectors, target, test_size=test_size, random_state=42)

        if algorithm == "LDA":
            try:

                lda = LinearDiscriminantAnalysis(n_components=dimension)
                lda.fit(X_train, y=y_train)
                reduced_vectors = lda.transform(vectors)
            except Exception as e:
                print(e)
                reduced_vectors = np.array(np.zeros(dimension))

        elif algorithm == "UMAP_SUPERVISED":
            umap_args = {'n_neighbors': 15,
                         'n_components': dimension,
                         'metric': 'cosine'}
            reducer = umap.UMAP(**umap_args).fit(X_train, y_train)
            reduced_vectors = reducer.transform(vectors)

        elif algorithm == "PARAMETRIC_UMAP_SUPERVISED":
            umap_args = {'n_neighbors': 15,
                         'n_components': dimension,
                         'metric': 'cosine',
                         "batch_size": batch_size,
                         "keras_fit_kwargs":{"verbose":1}}
            reducer = umap.parametric_umap.ParametricUMAP(**umap_args).fit(X_train, y_train)
            reduced_vectors = reducer.transform(vectors)

        elif algorithm == "UMAP_SUPERVISED_RNN":
            # Construct the RNN with Attention (Hyperparams identified through parameter optimization with OpTuna)
            encoder = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=dims),
                tf.keras.layers.Lambda(lambda x: tf.keras.backend.expand_dims(x, 1)),
                tf.keras.layers.Bidirectional(tf.keras.layers.GRU(1192, activation="relu", return_sequences=True)),
                attention(return_sequences=True),
                tf.keras.layers.Bidirectional(tf.keras.layers.GRU(933, activation="tanh", return_sequences=True)),
                attention(return_sequences=True),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1340),
                tf.keras.layers.Dense(units=dimension),
            ])
            umap_args = {'n_neighbors': 15,
                         'n_components': dimension,
                         'metric': 'cosine',
                         "encoder": encoder,
                         "dims": dims,
                         "batch_size": batch_size,
                         "keras_fit_kwargs":{"verbose":1}}
            reducer = umap.parametric_umap.ParametricUMAP(**umap_args).fit(X_train, y=y_train)
            reduced_vectors = reducer.transform(vectors)

        elif algorithm == "UMAP_SUPERVISED_TRANSFORMER":
            # Construct the Transformer Model (Hyperparams identified through parameter optimization with OpTuna)
            encoder = build_model(
                (dims),
                head_size=16,
                num_heads=16,
                ff_dim=256,
                num_transformer_blocks=2,
                mlp_units=[256, 128],
                mlp_dropout=0.0,
                dropout=0.0,
                n_classes=dimension
            )
            umap_args = {'n_neighbors': 15,
                         'n_components': dimension,
                         'metric': 'cosine',
                         "encoder": encoder,
                         "dims": dims,
                         "batch_size": batch_size,
                         "keras_fit_kwargs":{"verbose":0}}
            reducer = umap.parametric_umap.ParametricUMAP(**umap_args).fit(X_train, y=y_train)
            reduced_vectors = reducer.transform(vectors)

    return reduced_vectors


def load_dataset(dataset_name):
    if dataset_name == "20newsgroups":
        bert_vectors = np.array(joblib.load("resources/20Newsgroups_BERT.pkl"))
        data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
        target = data.target
    elif dataset_name == "trec6":
        bert_vectors = np.array(joblib.load("resources/trec_BERT.pkl"))
        data = pd.read_csv("data/trec.csv")
        target = data["label-coarse"]
    elif dataset_name == "trec50":
        bert_vectors = np.array(joblib.load("resources/trec_BERT.pkl"))
        data = pd.read_csv("data/trec.csv")
        target = data["label-fine"]
    elif dataset_name == "agnews":
        bert_vectors = np.array(joblib.load("resources/agnews_BERT.pkl"))
        train = pd.read_csv("data/ag_news/train.csv", header=None)
        test = pd.read_csv("data/ag_news/test.csv", header=None)
        data = pd.concat([train, test])
        target = data[0]
    return target, bert_vectors

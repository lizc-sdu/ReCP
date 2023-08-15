import json
import pickle
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import linear_model
import matplotlib.pyplot as plt

from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import f1_score

popus = np.load("./model_data/popus_count.npy")
cd = json.load(open("./model_data/mh_cd.json"))

cd_labels = np.zeros((len(cd)))
for i in range(len(cd)):
    cd_labels[i] = cd[str(i)]

n_clusters = 29


def regression(X_train, y_train, X_test, alpha):
    reg = linear_model.Ridge(alpha=alpha)
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)
    return y_pred


def kf_predict(X, Y):
    kf = KFold(n_splits=5)
    y_preds = []
    y_truths = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        y_pred = regression(X_train, y_train, X_test, 1)
        y_preds.append(y_pred)
        y_truths.append(y_test)

    return np.concatenate(y_preds), np.concatenate(y_truths)


def compute_metrics(y_pred, y_test):
    y_pred[y_pred < 0] = 0

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mae, np.sqrt(mse), r2


def predict_popus(emb):
    y_pred, y_test = kf_predict(emb, popus)
    mae, rmse, r2 = compute_metrics(y_pred, y_test)

    print('Popularity Prediction')
    print("MAE: ", mae)
    print("RMSE: ", rmse)
    print("R2: ", r2)

    return dict({'MAE': mae, 'RMSE': rmse, 'R2': r2})


def F_meansure(cd_labels, emb_labels):
    zones = len(cd_labels)

    labels = []
    preds = []
    for _i in range(zones):
        for _j in range(_i + 1, zones):
            cd_i, cd_j = cd_labels[_i], cd_labels[_j]
            emb_i, emb_j = emb_labels[_i], emb_labels[_j]

            if cd_i == cd_j:
                labels.append(1)
            else:
                labels.append(0)

            if emb_i == emb_j:
                preds.append(1)
            else:
                preds.append(0)

    bins = np.array([0, 0.5, 1])
    tn, fp, fn, tp = plt.hist2d(labels, preds, bins=bins, cmap='Blues')[0].flatten()

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    beta = 0.5
    f_beta = ((beta ** 2 + 1) * precision * recall) / (beta ** 2 * precision + recall)
    return f_beta


def lu_classify(emb):
    kmeans = KMeans(n_clusters=n_clusters, random_state=3)
    emb_labels = kmeans.fit_predict(emb)

    print('Land Usage Clustering')
    nmi = normalized_mutual_info_score(cd_labels, emb_labels)
    print("emb nmi: {:.3f}".format(nmi))
    ars = adjusted_rand_score(cd_labels, emb_labels)
    print("emb ars: {:.3f}".format(ars))
    f_score = F_meansure(cd_labels, emb_labels)
    print("emb f_score: {:.3f}".format(f_score))

    return dict({'NMI': nmi, 'ARS': ars, 'F_score': f_score})

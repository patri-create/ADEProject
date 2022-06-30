import itertools
import os
from pathlib import Path
import yaml
import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from pycm import *
from matplotlib import cm

from utils.data import get_project_root


def get_meddra_text(meddra_file, ptid_list):
    meddra_list = []
    meddra_dict = {}
    for line in open(meddra_file, 'r'):
        elems = line.split("$")
        ptid, text = int(elems[0]), elems[1]
        meddra_dict[ptid] = text

    for ptid in ptid_list:
        meddra_list.append(meddra_dict.get(ptid, ''))

    return meddra_list


def print_large_confusion_matrix(cm):
    confmat = np.random.rand(269, 269)
    ticks = np.linspace(0, 281, num=282)
    plt.figure(figsize=(15, 15))
    plt.imshow(cm.astype('float'), interpolation='none')
    plt.colorbar()
    plt.xticks(ticks, fontsize=4)
    plt.yticks(ticks, fontsize=4)
    plt.grid(False)
    plt.show()


def normalizer_evaluation_balanced(test_file, meddra_file, label, meddra, score_file, output, html_output, cm_output):
    # loading data, test_df[idmeddra] as truth and test_pred_labels as our predictions
    test_df = pd.read_csv(test_file)

    # getting the different possible classifications
    PRESENT_LABELS = sorted(test_df[label].unique())

    # loading scores and selecting the right prediction
    test_pred_scores = np.loadtxt(score_file)
    test_pred_labels = [PRESENT_LABELS[i] for i in test_pred_scores.argmax(axis=1)]

    # saving the predictions with their data
    test_df_copy = test_df.copy()
    test_df_copy[label] = test_pred_labels
    meddra_list = get_meddra_text(meddra_file, test_pred_labels)
    test_df_copy[meddra] = meddra_list
    test_df_copy.to_csv(output, index=False)

    # calculating the scores
    scores(test_df[label], test_pred_labels)

    # printing the results
    print(classification_report(test_df[label], test_pred_labels, zero_division=0))

    # creating confusion matrix with pycm
    conf_matrix = ConfusionMatrix(test_df[label].values, test_pred_labels, classes=PRESENT_LABELS)
    # output in html format
    conf_matrix.save_html(html_output, color=(100, 50, 250))

    # creating and saving cm
    conf_matrix = confusion_matrix(test_df[label], test_pred_labels)
    plot_confusion_matrix(cm=conf_matrix, classes=PRESENT_LABELS, png_output=cm_output, show=False)


def normalizer_evaluation(master_file, test_file, meddra_file, label, meddra, score_file, output, html_output,
                          cm_output):
    # loading data, test_df[idmeddra] as truth and test_pred_labels as our predictions
    master_df = pd.read_csv(master_file)
    test_df = pd.read_csv(test_file)

    # getting the different possible classifications
    PRESENT_LABELS = sorted(master_df[label].unique())

    # loading scores and selecting the right prediction
    test_pred_scores = np.loadtxt(score_file)
    test_pred_labels = [PRESENT_LABELS[i] for i in test_pred_scores.argmax(axis=1)]

    # saving the predictions with their data
    test_df_copy = test_df.copy()
    test_df_copy[label] = test_pred_labels
    meddra_list = get_meddra_text(meddra_file, test_pred_labels)
    test_df_copy[meddra] = meddra_list
    test_df_copy.to_csv(output, index=False)

    # creating confusion matrix with pycm
    t = set(test_df[label])
    p = set(test_pred_labels)
    PRESENT_LABELS = list(t | p)
    conf_matrix = ConfusionMatrix(test_df[label].values, test_pred_labels, classes=PRESENT_LABELS)

    # output in html format
    conf_matrix.save_html(html_output, color=(100, 50, 250))

    # calculating the scores
    scores(test_df[label], test_pred_labels)
    print(classification_report(test_df[label], test_pred_labels, labels=np.unique(test_pred_labels), zero_division=0))

    # printing and output cm
    conf_matrix = confusion_matrix(test_df[label], test_pred_labels)
    plot_confusion_matrix(cm=conf_matrix, classes=PRESENT_LABELS, png_output=cm_output, show=False)


def normalizer_sim_evaluation(test_file, label, meddra, score_file, output, html_output, cm_output):
    # loading data, test_df[idmeddra] as truth and test_pred_labels as our predictions
    test_df = pd.read_csv(test_file)

    # loading scores and selecting the right prediction
    test_pred_scores = pd.read_csv(score_file)
    test_pred_labels = test_pred_scores['idmeddra'].values

    # saving the predictions with their data
    test_df_copy = test_df.copy()
    test_df_copy[label] = test_pred_labels
    test_df_copy[meddra] = test_pred_scores['meddra']
    test_df_copy.to_csv(output, index=False)

    # creating confusion matrix with pycm
    conf_matrix = ConfusionMatrix(test_df[label].values, test_pred_labels)

    # output in html format
    conf_matrix.save_html(html_output, color=(100, 50, 250))

    # calculating the scores
    scores(test_df[label], test_pred_labels)

    # printing the results
    print(classification_report(test_df[label], test_pred_labels, zero_division=0))
    # print_large_confusion_matrix(confusion_matrix(sorted(test_df[label]), sorted(test_pred_labels)))

    # printing and output cm
    v = test_df[label].sort_values()
    v2 = np.sort(test_pred_labels)

    t = set(test_df[label])
    p = set(test_pred_labels)
    PRESENT_LABELS = np.sort(list(t | p))
    conf_matrix = confusion_matrix(test_df[label], test_pred_labels)
    plot_confusion_matrix(cm=conf_matrix, classes=PRESENT_LABELS, png_output=cm_output, show=False)


def plot_confusion_matrix(cm, classes, normalize=True, cmap=cm.Blues, png_output="output/", show=True):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = 'Normalized confusion matrix'
    else:
        title = 'Confusion matrix'

    # Calculate chart area size
    leftmargin = 0.5  # inches
    rightmargin = 0.5  # inches
    categorysize = 0.5  # inches
    figwidth = leftmargin + rightmargin + (len(classes) * categorysize)

    f = plt.figure(figsize=(figwidth, figwidth))

    # Create an axes instance and ajust the subplot size
    ax = f.add_subplot(111)
    ax.set_aspect(1)
    f.subplots_adjust(left=leftmargin / figwidth, right=1 - rightmargin / figwidth, top=0.94, bottom=0.1)

    res = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)
    plt.colorbar(res)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticklabels(classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if png_output is not None:
        f.savefig(os.path.join(png_output, 'confusion_matrix.png'), bbox_inches='tight')

    if show:
        plt.show()
        plt.close(f)
    else:
        plt.close(f)


def print_confusion_matrix(confusion_matrix, class_names, figsize=(10, 7), fontsize=14):
    confusion_matrix_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1, keepdims=True)
    df_cm = pd.DataFrame(
        confusion_matrix_normalized, index=class_names, columns=class_names,
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt=".2f")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('Truth')
    plt.xlabel('Prediction')
    plt.show()


def classifier_and_extractor_evaluation(test_file, label, score_file, output, extractor=False):
    # loading data, test_df[category] as truth and test_pred_labels as our predictions
    test_df = pd.read_csv(test_file)

    # getting the different possible classifications
    PRESENT_LABELS = sorted(test_df[label].unique())

    # loading scores and selecting the right prediction
    test_pred_scores = np.loadtxt(score_file)
    test_pred_labels = [PRESENT_LABELS[i] for i in test_pred_scores.argmax(axis=1)]

    # saving the predictions with their data
    test_df_copy = test_df.copy()
    test_df_copy[label] = test_pred_labels
    test_df_copy.to_csv(output, index=False)

    # calculating the scores
    scores(test_df[label], test_pred_labels)

    # creating confusion matrix
    conf_matrix = confusion_matrix(test_df[label], test_pred_labels)

    # printing the results
    print(classification_report(test_df[label], test_pred_labels))
    print_confusion_matrix(conf_matrix, PRESENT_LABELS)

    if extractor:
        extractor_evaluation(test_df, label, test_pred_labels)


def extractor_evaluation(test_df, label, test_pred_labels):
    # converting I-ADE and B-ADE to ADE
    PRESENT_LABELS = ['ADE', 'O']
    test_df.loc[test_df[label] == 'B-ADE', label] = 'ADE'
    test_df.loc[test_df[label] == 'I-ADE', label] = 'ADE'
    test_pred_labels = ['ADE' if i == 'B-ADE' or i == 'I-ADE' else i for i in test_pred_labels]

    # calculating the scores
    scores(test_df[label], test_pred_labels)

    # creating confusion matrix
    conf_matrix = confusion_matrix(test_df[label], test_pred_labels)

    # printing the results
    print(classification_report(test_df[label], test_pred_labels))
    print_confusion_matrix(conf_matrix, PRESENT_LABELS)


def scores(truth, prediction):
    f1_score(y_true=truth, y_pred=prediction, average="macro"),
    f1_score(y_true=truth, y_pred=prediction, average="micro")


def evaluate(model):
    # loading config params
    project_root: Path = get_project_root()
    with open(str(project_root / "config.yml")) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    if model == 'classifier':
        test_file = params["classifier_data"]["test"]
        label = params["classifier_data"]["label_field"]
        score_file = params["classifier_data"]["test_pred_scores"]
        output = params["classifier_data"]["test_pred"]
        classifier_and_extractor_evaluation(test_file, label, score_file, output)
    elif model == 'extractor':
        test_file = params["extractor_data"]["iob2_test"]
        label = params["extractor_data"]["label_field"]
        score_file = params["extractor_data"]["test_pred_scores"]
        output = params["extractor_data"]["test_pred"]
        classifier_and_extractor_evaluation(test_file, label, score_file, output, True)
    elif model == 'normalizer':
        master_file = params["normalizer_data"]["master"]
        test_file = params["normalizer_data"]["test"]
        meddra_file = params['normalizer_data']['meddra']
        label = params["normalizer_data"]["label_field"]
        meddra = params["normalizer_data"]["meddra_field"]
        score_file = params["normalizer_data"]["test_pred_scores"]
        output = params["normalizer_data"]["test_pred"]
        html_output = params["normalizer_data"]["html"]
        cm_output = params["normalizer_data"]["cm_output"]

        # normalizer_evaluation(master_file, test_file, meddra_file, label, meddra, score_file, output, html_output,
        #                       cm_output)

        normalizer_evaluation_balanced(test_file, meddra_file, label, meddra, score_file, output, html_output, cm_output)

    elif model == 'similarity':
        test_file = params["normalizer_data"]["test"]
        label = params["normalizer_data"]["label_field"]
        meddra = params["normalizer_data"]["meddra_field"]
        output = params["normalizer_data"]["test_pred_sim"]
        score_file = params["normalizer_data"]["test_pred_scores_sim"]
        html_output = params["normalizer_data"]["html_sim"]
        cm_output = params["normalizer_data"]["cm_output"]

        normalizer_sim_evaluation(test_file, label, meddra, score_file, output, html_output, cm_output)


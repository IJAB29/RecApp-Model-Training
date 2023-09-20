# """stem or lemmatize the thingies"""
import stws
from salitaulap import *
from stws import TfMonoVectorizer
from stws import TfEMonoVectorizer
from stws import TfIgmVectorizer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import matplotlib.pyplot as plt
CSV_DIR = r"ALL_DATA(oversampled).csv"

docs = pd.read_csv(CSV_DIR, index_col=False)
background = np.array(docs["Backgrounds"])
category = np.array(docs["Categories"])
abstract = np.array(docs["Abstracts"])

# results = {"Dataset": [],
#            "Model": [],
#            "F1 Score": []}

ab_scores = {"Model": [],
             "F1 Score": []}

bg_scores = {"Model": [],
             "F1 Score": []}


def createWeightsCsv():
    mono_bg = TfMonoVectorizer(doc_text=background, doc_label=category, alpha=6).getMONOGlobal()

    emono_bg = TfEMonoVectorizer(doc_text=background, doc_label=category, alpha=6).getEMONOGlobal()

    igm_bg = TfIgmVectorizer(doc_text=background, doc_label=category, alpha=6).getIGMGlobal()

    mono_ab = TfMonoVectorizer(doc_text=abstract, doc_label=category, alpha=6).getMONOGlobal()

    emono_ab = TfEMonoVectorizer(doc_text=abstract, doc_label=category, alpha=6).getEMONOGlobal()

    igm_ab = TfIgmVectorizer(doc_text=abstract, doc_label=category, alpha=6).getIGMGlobal()

    df_bg = pd.DataFrame({"MONO": mono_bg[0], "EMONO": emono_bg[0], "IGM": igm_bg[0]}, index=mono_bg[1])
    df_bg.to_csv("BG_WEIGHTS.csv")
    df_ab = pd.DataFrame({"MONO": mono_ab[0], "EMONO": emono_ab[0], "IGM": igm_ab[0]}, index=mono_ab[1])
    df_ab.to_csv("AB_WEIGHTS.csv")

    # df_mono_bg = pd.DataFrame({"Terms": mono_bg[1], "Weights": mono_bg[0]}).to_csv("BG MONO WEIGHTS.csv", index=False)
    # df_emono_bg = pd.DataFrame({"Terms": emono_bg[1], "Weights": emono_bg[0]}).to_csv("BG EMONO WEIGHTS.csv", index=False)
    # df_igm_bg = pd.DataFrame({"Terms": igm_bg[1], "Weights": igm_bg[0]}).to_csv("BG IGM WEIGHTS.csv", index=False)

    # df_mono_ab = pd.DataFrame({"Terms": mono_ab[1], "Weights": mono_ab[0]}).to_csv("AB MONO WEIGHTS.csv", index=False)
    # df_emono_ab = pd.DataFrame({"Terms": emono_ab[1], "Weights": emono_ab[0]}).to_csv("AB EMONO WEIGHTS.csv", index=False)
    # df_igm_ab = pd.DataFrame({"Terms": igm_ab[1], "Weights": igm_ab[0]}).to_csv("AB IGM WEIGHTS.csv", index=False)


def createClassifier(data, vectorizer, stws, label):
    name = f"{data} {vectorizer}"
    X_train, X_test, y_train, y_test = train_test_split(stws, label, test_size=0.30, random_state=42)
    knn = KNeighborsClassifier().fit(X_train, y_train)
    nvb = MultinomialNB().fit(X_train, y_train)
    # svc = SVC(kernel="linear", probability=True).fit(X_train, y_train)
    linear_svc = LinearSVC().fit(X_train, y_train)

    dictionary = CountVectorizer(vocabulary=stws)
    pickle.dump(dictionary, open(f"pickles/Dictionary {name}.pkl", "wb"))

    # pd.DataFrame(stws).to_csv(f"{name}.csv")

    evaluateClassifier(data, vectorizer, X_test, y_test, knn)
    evaluateClassifier(data, vectorizer, X_test, y_test, nvb)
    # evaluateClassifier(data, vectorizer, X_test, y_test, svc)
    evaluateClassifier(data, vectorizer, X_test, y_test, linear_svc)


def evaluateClassifier(data, vectorizer, X_test, y_test, classifier):
    name = f"{data} {vectorizer}"
    pickle.dump(classifier, open(f"pickles/{name} {type(classifier).__name__}.pkl", "wb"))
    pred_y = classifier.predict(X_test)

    # micro = metrics.f1_score(y_test, pred_y, average='micro')
    # macro = metrics.f1_score(y_test, pred_y, average='macro')
    # weighted = metrics.f1_score(y_test, pred_y, average='weighted')
    # print(f"\n{stws_name}\t{type(classifier).__name__}\t{micro}\t{macro}\t{weighted}")

    precision = metrics.precision_score(y_true=y_test, y_pred=pred_y, average=None)
    recall = metrics.recall_score(y_true=y_test, y_pred=pred_y, average=None)
    accuracy = metrics.accuracy_score(y_true=y_test, y_pred=pred_y)
    f1 = metrics.f1_score(y_true=y_test, y_pred=pred_y, average='micro')
    clf_name = type(classifier).__name__
    model = f"{vectorizer} {clf_name}"
    print(f"\n{name} {type(classifier).__name__}\tpre: {precision}\trec: {recall}\tacc: {accuracy}\tf1: {f1}")
    # print(f"\n{name} {clf_name}\t F1 Score: {f1}")
    if data == "ABSTRACT":
        ab_scores["Model"].append(model)
        ab_scores["F1 Score"].append(f1)
    elif data == "BACKGROUND":
        bg_scores["Model"].append(model)
        bg_scores["F1 Score"].append(f1)


tfmono_bg = TfMonoVectorizer(doc_text=background, doc_label=category, alpha=6).getTF_MONO()
sqrttfmono_bg = TfMonoVectorizer(doc_text=background, doc_label=category, alpha=6, sqrt=True).getTF_MONO()

tfemono_bg = TfEMonoVectorizer(doc_text=background, doc_label=category, alpha=6).getTF_EMONO()
sqrttfemono_bg = TfEMonoVectorizer(doc_text=background, doc_label=category, alpha=6, sqrt=True).getTF_EMONO()

tfigm_bg = TfIgmVectorizer(doc_text=background, doc_label=category, alpha=6).getTF_IGM()
sqrttfigm_bg = TfIgmVectorizer(doc_text=background, doc_label=category, alpha=6, sqrt=True).getTF_IGM()

tfmono_ab = TfMonoVectorizer(doc_text=abstract, doc_label=category, alpha=6).getTF_MONO()
sqrttfmono_ab = TfMonoVectorizer(doc_text=abstract, doc_label=category, alpha=6, sqrt=True).getTF_MONO()

tfemono_ab = TfEMonoVectorizer(doc_text=abstract, doc_label=category, alpha=6).getTF_EMONO()
sqrttfemono_ab = TfEMonoVectorizer(doc_text=abstract, doc_label=category, alpha=6, sqrt=True).getTF_EMONO()

tfigm_ab = TfIgmVectorizer(doc_text=abstract, doc_label=category, alpha=6).getTF_IGM()
sqrttfigm_ab = TfIgmVectorizer(doc_text=abstract, doc_label=category, alpha=6, sqrt=True).getTF_IGM()

createClassifier(data="BACKGROUND", vectorizer="TF MONO", stws=tfmono_bg, label=category)
createClassifier(data="BACKGROUND", vectorizer="SQRT TF MONO", stws=sqrttfmono_bg, label=category)
createClassifier(data="BACKGROUND", vectorizer="TF EMONO", stws=tfemono_bg, label=category)
createClassifier(data="BACKGROUND", vectorizer="SQRT TF EMONO", stws=sqrttfemono_bg, label=category)
createClassifier(data="BACKGROUND", vectorizer="TF IGM", stws=tfigm_bg, label=category)
createClassifier(data="BACKGROUND", vectorizer="SQRT TF IGM", stws=sqrttfigm_bg, label=category)

createClassifier(data="ABSTRACT", vectorizer="TF MONO", stws=tfmono_ab, label=category)
createClassifier(data="ABSTRACT", vectorizer="SQRT TF MONO", stws=sqrttfmono_ab, label=category)
createClassifier(data="ABSTRACT", vectorizer="TF EMONO", stws=tfemono_ab, label=category)
createClassifier(data="ABSTRACT", vectorizer="SQRT TF EMONO", stws=sqrttfemono_ab, label=category)
createClassifier(data="ABSTRACT", vectorizer="TF IGM", stws=tfigm_ab, label=category)
createClassifier(data="ABSTRACT", vectorizer="SQRT TF IGM", stws=sqrttfigm_ab, label=category)

# createWeightsCsv()
#
# ab_features = stws.getFeatures(abstract, category)
# bg_features = stws.getFeatures(background, category)
# salitaulap.generateWordCloud(ab_features)
# salitaulap.generateWordCloud(bg_features)
# ab_features.to_csv(f"AB Features.csv")
# bg_features.to_csv(f"BG Features.csv")
#
visualize(ab_scores, "Model", "F1 Score")
visualize(bg_scores, "Model", "F1 Score")
#
# pd.DataFrame(ab_scores).to_csv("AB Scores.csv", index=False)
# pd.DataFrame(bg_scores).to_csv("BG Scores.csv", index=False)
"""-----------------------------------------Predicting New-----------------------------------------"""


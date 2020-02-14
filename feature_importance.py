import pandas as pd
import random
import math
import os
import numpy as np
from imblearn.over_sampling import SMOTE
from itertools import combinations
from sklearn.neighbors.kde import KernelDensity
from sklearn.feature_extraction import DictVectorizer
from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import GenericUnivariateSelect, chi2, f_classif, VarianceThreshold
from sklearn.feature_selection import SelectFpr
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import RepeatedKFold
from sklearn.neural_network import MLPClassifier
from operator import itemgetter
import csv
from sklearn.model_selection import RepeatedKFold
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.svm import SVR
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import GenericUnivariateSelect, chi2, f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectFpr
from sklearn.feature_selection import SelectFdr
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.externals import joblib
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC, SVR
from sklearn.model_selection import RepeatedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import GenericUnivariateSelect, chi2, f_classif, VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectFpr
from sklearn.feature_selection import SelectFdr
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import mutual_info_classif
# import matplotlib.pyplot as plt
from sklearn import metrics
import time
import csv
import pickle
# import pyswarms as ps
from sklearn.neural_network import MLPClassifier
import statistics
import itertools
from sklearn.cluster import KMeans
from skfeature.function.information_theoretical_based import CMIM, JMI, MIM, MRMR, RJMI, RelaxMRMR
import warnings
import time

# import matplotlib
# matplotlib.use('TkAgg')


def main():
    path = os.getcwd()
    start_time = time.time()
    feat_sel_no_of_splits = 5
    feat_sel_no_of_repeats = 10
    no_of_top_features_for_feat_sel = 15
    classifier_no_of_splits = 5
    classifier_no_of_repeats = 10
    max_num_of_features = 16
    test_file = 'non_imputed_full_test_data.csv'
    train_file = ''
    model_directory = ''

    # imputation = 'none'
    # imputation = 'wknn'
    # is_somte = 'no'
    # is_somte = 'yes'
    # data_balanced = 'no'
    data_balanced = 'yes'
    dataset_comb_list = [('none', 'no'), ('wknn', 'no'), ('none', 'yes'), ('wknn', 'yes')]

    for imputation, is_somte in dataset_comb_list:
        print("--Running imputation:", imputation, " smote:", is_somte)
        if imputation=='none' and is_somte == 'no':
            train_file = 'non_imputed_full_data.csv'
            model_directory = 'non_impute_feature_imp_results'
        if imputation == 'none' and is_somte == 'yes':
            train_file = 'non_imputed_full_data.csv'
            model_directory = 'non_impute_feature_imp_smote_results'
        if imputation == 'wknn' and is_somte == 'no':
            train_file = 'weighted_knn_imputed_data.csv'
            model_directory = 'weighted_knn_feature_imp'
        if imputation == 'wknn' and is_somte == 'yes':
            train_file = 'weighted_knn_imputed_data.csv'
            model_directory = 'weighted_knn_smote_feature_imp'

        # model_result_directory = path + '\\' + model_directory + '\\'
        # model_feat_imp_directory = path + '\\' + model_directory + '\\feature_importance\\'
        model_result_directory = os.path.join(path, model_directory)
        model_feat_imp_directory = os.path.join(path, model_directory, "feature_importance")


        feature_selection_method_list = ['SelectFpr', 'SelectFdr', 'mutual_info_classif', 'feat_importance', 'L1_based',
                                         'selectkbest_f_classif', 'GenericUnivariateSelect', 'SelectPercentile',
                                         'VarianceThreshold', 'conditional_mutual_info_maximisation', 'joint_mutual_info',
                                         'mutual_info_maximisation', 'max_relevance_min_redundancy']
        # feature_selection_method_list = ['SelectFpr']
        # classifier_method_list = ['RDMF']
        classifier_method_list = ['SVM', 'RDMF', 'MLP']
        X, Y = readDataFromFile(train_file, test_file, is_somte)
        for feature_selection_method in feature_selection_method_list:
            top_10_features = repeated_feature_selection(X, Y, feature_selection_method, feat_sel_no_of_splits,
                                                         feat_sel_no_of_repeats, no_of_top_features_for_feat_sel, model_feat_imp_directory)
            print("top 10 ", feature_selection_method, " features - ", top_10_features)
            for classifier_method in classifier_method_list:
                feature_selection_auc_result, feature_selection_accuracy_result, feature_selection_result = feature_combination_classifier(X, Y, classifier_no_of_splits, classifier_no_of_repeats,
                                                                          top_10_features, classifier_method, max_num_of_features)
                # feature_selection_auc_result.to_csv(feature_selection_method+"-"+classifier_method+"-AUC.csv")
                # feature_selection_accuracy_result.to_csv(feature_selection_method + "-" + classifier_method + "-ACCURACY.csv")
                # feature_selection_result.to_csv(feature_selection_method + "-" + classifier_method + ".csv")
                feature_selection_result.to_csv(os.path.join(model_result_directory, feature_selection_method + "-" + classifier_method + ".csv"))
                # feature_selection_result.to_csv(
                #     model_result_directory + feature_selection_method + "-" + classifier_method + ".csv")
        print("-done")
    print("total time taken - ", time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time)))


def feature_combination_classifier(X, Y, n_splits, n_repeats, top_10_features, method, max_num_of_features):
    result_auc_columns = ["feature combination"]
    result_accuracy_columns = ["feature combination"]
    for i in range(1, (n_splits * n_repeats) + 1):
        result_auc_columns.append("AUC-" + str(i))
        result_accuracy_columns.append("ACCURACY-" + str(i))
    # print(result_columns)
    result_columns = result_auc_columns + result_accuracy_columns
    auc_result_dataframe = pd.DataFrame(columns=result_auc_columns)
    accuracy_result_dataframe = pd.DataFrame(columns=result_accuracy_columns)
    result_dataframe = pd.DataFrame(columns=result_columns)
    for i in range(1, max_num_of_features):
        repeated_classifier_auc_result_list, repeated_classifier_accuracy_result_list = repeated_classifier_method(X, Y, n_splits, n_repeats, top_10_features[:i],
                                                                         method)
        repeated_classifier_result_list = repeated_classifier_auc_result_list + repeated_classifier_accuracy_result_list
        # print(repeated_classifier_result_list)
        tem_auc_df = pd.DataFrame([repeated_classifier_auc_result_list], columns=result_auc_columns)
        tem_accuracy_df = pd.DataFrame([repeated_classifier_accuracy_result_list], columns=result_accuracy_columns)
        tem_df = pd.DataFrame([repeated_classifier_result_list], columns=result_columns)
        auc_result_dataframe = auc_result_dataframe.append(tem_auc_df, ignore_index=True)
        accuracy_result_dataframe = accuracy_result_dataframe.append(tem_accuracy_df, ignore_index=True)
        result_dataframe = result_dataframe.append(tem_df, ignore_index=True)
        # print(auc_result_dataframe)
        # print(accuracy_result_dataframe)
        # print(result_dataframe)
    return auc_result_dataframe, accuracy_result_dataframe, result_dataframe


def repeated_classifier_method(X_input, Y, n_splits, n_repeats, feature_combination, method):
    X_feature_select = get_data_from_column_list(X_input, feature_combination)
    # print(X_feature_select)
    # scaler = StandardScaler()
    # x_scaled = scaler.fit_transform(X_feature_select)
    # X = pd.DataFrame(x_scaled, index=X_feature_select.index, columns=X_feature_select.columns)
    X = X_feature_select
    repeated_classifier_auc_result_list = [feature_combination]
    repeated_classifier_accuracy_result_list = [feature_combination]
    kf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=None)
    for train_index, test_index in kf.split(X):
        x_train, x_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        if method == "MLP":
            classifier = MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=200)
        if method == "SVM":
            classifier = SVC(gamma='auto', probability=True)
        if method == "RDMF":
            classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=42)
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        # rfr_prediction_proba = classifier.predict_proba(x_test)[:, 0]
        auc, accuracy = evaluation_metrics(y_test, y_pred)
        repeated_classifier_auc_result_list.append(auc)
        repeated_classifier_accuracy_result_list.append(accuracy)
    return repeated_classifier_auc_result_list, repeated_classifier_accuracy_result_list


def evaluation_metrics(y_test, y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=None)
    auc = metrics.auc(fpr, tpr)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    return auc, accuracy


def repeated_feature_selection(X, Y, feature_method, n_splits, n_repeats, no_of_top_features, model_feat_imp_directory):
    feature_frequency = {}
    kf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=None)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
        X_train = standard_scalar(X_train)
        top_20_feature = []
        if feature_method == "SelectFpr":
            top_20_feature = feature_SelectFpr(X_train, y_train)
        if feature_method == "mutual_info_classif":
            top_20_feature = feature_mutual_info_classif(X_train, y_train)
        if feature_method == "feat_importance":
            top_20_feature = FeatureImportance(X_train, y_train)
        if feature_method == "L1_based":
            top_20_feature = L1_based_feature_selection(X_train, y_train)
        if feature_method == "SelectFdr":
            top_20_feature = feature_SelectFdr(X_train, y_train)
        if feature_method == "selectkbest_f_classif":
            top_20_feature = feature_selectkbest_f_classif(X_train, y_train)
        if feature_method == "selectkbest_chi2":
            top_20_feature = feature_selectkbest_chi2(X_train, y_train)
        if feature_method == "GenericUnivariateSelect":
            top_20_feature = feature_GenericUnivariateSelect(X_train, y_train)
        if feature_method == "SelectPercentile":
            top_20_feature = feature_SelectPercentile(X_train, y_train)
        if feature_method == "feature_RFE":
            top_20_feature = feature_RFE(X_train, y_train)
        if feature_method == "feature_RFECV":
            top_20_feature = feature_RFECV(X_train, y_train)
        if feature_method == "VarianceThreshold":
            top_20_feature = feature_VarianceThreshold(X_train, y_train)
        if feature_method == "conditional_mutual_info_maximisation":
            top_20_feature = feature_conditional_mutual_info_maximisation(X_train, y_train)
        if feature_method == "joint_mutual_info":
            top_20_feature = feature_joint_mutual_info(X_train, y_train)
        if feature_method == "mutual_info_maximisation":
            top_20_feature = feature_mutual_info_maximisation(X_train, y_train)
        if feature_method == "max_relevance_min_redundancy":
            top_20_feature = feature_max_relevance_min_redundancy(X_train, y_train)
        for rowindex, rowitem in top_20_feature.iterrows():
            # feature_frequency[rowitem['Specs']] = rowitem['Score']
            if rowitem['Specs'] in feature_frequency:
                feature_frequency[rowitem['Specs']] += rowitem['Score']
            else:
                feature_frequency[rowitem['Specs']] = rowitem['Score']
    # print(feature_frequency)
    feature_frequency = {k: v / (n_splits * n_repeats) for k, v in feature_frequency.items()}
    # print(feature_frequency)
    feature_frequency = sorted(feature_frequency.items(), key=itemgetter(1), reverse=True)
    # print(feature_frequency)
    no_of_features = 1
    top_features = []
    feature_importance_dataframe = pd.DataFrame(feature_frequency, columns=['feature', 'feature importance value'])
    feature_importance_dataframe.to_csv(os.path.join(model_feat_imp_directory, feature_method + "-importance.csv"))
    # feature_importance_dataframe.to_csv(model_feat_imp_directory + feature_method + "-importance.csv")
    # print(feature_importance_dataframe)
    for feature_tuple in feature_frequency:
        if no_of_features <= no_of_top_features:
            top_features.append(feature_tuple[0])
            no_of_features += 1
    # print(top_features)
    return top_features

def standard_scalar(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return pd.DataFrame(scaled_data, index=data.index, columns=data.columns)

def feature_mutual_info_classif(x_data, y_data):
    feature_scores = mutual_info_classif(x_data, y_data)
    dfscores = pd.DataFrame(feature_scores)
    dfcolumns = pd.DataFrame(x_data.columns)
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    top_20_features = featureScores.nlargest(20, 'Score')
    return top_20_features


def feature_SelectFpr(x_data, y_data):
    # print(x_data)
    # print(y_data)
    bestfeatures = SelectFpr(f_classif, alpha=0.01)
    fit = bestfeatures.fit(x_data, y_data)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(x_data.columns)
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    top_20_features = featureScores.nlargest(20, 'Score')
    return top_20_features


def FeatureImportance(x_data, y_data):
    model = ExtraTreesClassifier()
    model.fit(x_data, y_data)
    dfscores = pd.DataFrame(model.feature_importances_)
    dfcolumns = pd.DataFrame(x_data.columns)
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    top_20_features = featureScores.nlargest(20, 'Score')
    return top_20_features


def L1_based_feature_selection(x_data, y_data):
    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(x_data, y_data)
    model = SelectFromModel(lsvc, prefit=True)
    dfscores = pd.DataFrame(model.get_support())
    dfcolumns = pd.DataFrame(x_data.columns)
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    top_20_features = featureScores.nlargest(20, 'Score')
    return top_20_features



def feature_SelectFdr(x_data, y_data):
    bestfeatures = SelectFdr(f_classif, alpha=0.01)
    fit = bestfeatures.fit(x_data, y_data)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(x_data.columns)
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    top_20_features = featureScores.nlargest(20, 'Score')
    return top_20_features


def feature_selectkbest_f_classif(x_data, y_data):
    bestfeatures = SelectKBest(score_func=f_classif, k=10)
    fit = bestfeatures.fit(x_data, y_data)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(x_data.columns)
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    top_20_features = featureScores.nlargest(20, 'Score')
    return top_20_features


def feature_selectkbest_chi2(x_data, y_data):
    # print("Running selectkbest chi2")
    bestfeatures = SelectKBest(score_func=chi2, k=10)
    fit = bestfeatures.fit(x_data, y_data)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(x_data.columns)
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    top_20_features = featureScores.nlargest(20, 'Score')
    return top_20_features


def feature_GenericUnivariateSelect(x_data, y_data):
    bestfeatures = GenericUnivariateSelect(f_classif, 'k_best', param=20)
    fit = bestfeatures.fit(x_data, y_data)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(x_data.columns)
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    top_20_features = featureScores.nlargest(20, 'Score')
    return top_20_features



def feature_SelectPercentile(x_data, y_data):
    bestfeatures = SelectPercentile(f_classif, percentile=10)
    fit = bestfeatures.fit(x_data, y_data)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(x_data.columns)
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    top_20_features = featureScores.nlargest(20, 'Score')
    return top_20_features


def feature_RFE(x_data, y_data):
    print("feature_RFE")
    estimator = SVR(kernel="linear")
    bestfeatures = RFE(estimator, 5, step=1)
    fit = bestfeatures.fit(x_data, y_data)
    dfscores = pd.DataFrame(fit.ranking_)
    dfcolumns = pd.DataFrame(x_data.columns)
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    top_20_features = featureScores.nlargest(20, 'Score')
    return top_20_features



def feature_RFECV(x_data, y_data):
    estimator = SVR(kernel="linear")
    bestfeatures = RFECV(estimator, step=1, cv=5)
    fit = bestfeatures.fit(x_data, y_data)
    dfscores = pd.DataFrame(fit.ranking_)
    dfcolumns = pd.DataFrame(x_data.columns)
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    top_20_features = featureScores.nlargest(20, 'Score')
    return top_20_features


def feature_VarianceThreshold(x_data, y_data):
    bestfeatures = VarianceThreshold(threshold=(.8 * (1 - .8)))
    fit = bestfeatures.fit(x_data, y_data)
    dfscores = pd.DataFrame(fit.get_support())
    dfcolumns = pd.DataFrame(x_data.columns)
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    top_20_features = featureScores.nlargest(20, 'Score')
    return top_20_features


def feature_conditional_mutual_info_maximisation(x_data, y_data):
    features_scores = CMIM.cmim(x_data.values, y_data.values, n_selected_features=20)
    features_index = [int(index[0]) for index in features_scores]
    feat_list = x_data.columns.values[features_index]
    feat_list_with_imp = [(feat_list[i], features_scores[i][1]) for i in range(len(features_scores))]
    # dfscores = pd.DataFrame(features_scores)
    # dfcolumns = pd.DataFrame(x_data.columns)
    # featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores = pd.DataFrame(feat_list_with_imp)
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    top_20_features = featureScores.nlargest(20, 'Score')
    return top_20_features


def feature_joint_mutual_info(x_data, y_data):
    features_scores = JMI.jmi(x_data.values, y_data.values, n_selected_features=20)
    features_index = [int(index[0]) for index in features_scores]
    feat_list = x_data.columns.values[features_index]
    feat_list_with_imp = [(feat_list[i], features_scores[i][1]) for i in range(len(features_scores))]
    # dfscores = pd.DataFrame(features_scores)
    # dfcolumns = pd.DataFrame(x_data.columns)
    # featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores = pd.DataFrame(feat_list_with_imp)
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    top_20_features = featureScores.nlargest(20, 'Score')
    return top_20_features


def feature_mutual_info_maximisation(x_data, y_data):
    features_scores = MIM.mim(x_data.values, y_data.values, n_selected_features=20)
    features_index = [int(index[0]) for index in features_scores]
    feat_list = x_data.columns.values[features_index]
    feat_list_with_imp = [(feat_list[i], features_scores[i][1]) for i in range(len(features_scores))]
    # dfscores = pd.DataFrame(features_scores)
    # dfcolumns = pd.DataFrame(x_data.columns)
    # featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores = pd.DataFrame(feat_list_with_imp)
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    top_20_features = featureScores.nlargest(20, 'Score')
    return top_20_features


def feature_max_relevance_min_redundancy(x_data, y_data):
    features_scores = MRMR.mrmr(x_data.values, y_data.values, n_selected_features=20)
    features_index = [int(index[0]) for index in features_scores]
    feat_list = x_data.columns.values[features_index]
    feat_list_with_imp = [(feat_list[i], features_scores[i][1]) for i in range(len(features_scores))]
    # dfscores = pd.DataFrame(features_scores)
    # dfcolumns = pd.DataFrame(x_data.columns)
    # featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores = pd.DataFrame(feat_list_with_imp)
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    top_20_features = featureScores.nlargest(20, 'Score')
    return top_20_features


def readDataFromFile(train_file, test_file, is_smote):
    train_data = pd.read_csv(train_file)
    target_data_map = {'stunting36': {'yes': 1, 'no': 0}}
    train_data.replace(target_data_map, inplace=True)
    target_data = train_data.iloc[:, train_data.columns.get_loc('stunting36')]
    test_data = pd.read_csv(test_file)
    cols_to_remove = ['mage_cat', 'mheight_cat', 'mbmi_cat_enroll', 'fbmi_cat', 'preterm', 'mbmi_6month_cat',
                      'finalgestdel_cat', 'fedu_cat', 'pregnum_cat', 'fjob_cat', 'mjob_cat', 'medu_cat', ]
    for col in train_data.columns.values:
        if "36" in col:
            cols_to_remove.append(col)
        if "HAZ" in col:
            cols_to_remove.append(col)
        if "inflength" in col:
            cols_to_remove.append(col)
        if "stunting" in col:
            cols_to_remove.append(col)
    cols_to_remove.remove('inflength_6month')
    after_drop_train_data = drop_data_from_column_list(train_data, cols_to_remove)
    one_hot_train_data = one_hot_encoding(after_drop_train_data)
    one_hot_test_data = one_hot_encoding(test_data)
    common_columns = list(set(one_hot_train_data.columns.values).intersection(one_hot_test_data.columns.values))
    dep_var_data = get_data_from_column_list(one_hot_train_data, common_columns)
    if is_smote == 'yes':
        X_smote, Y_smote = smote(dep_var_data, target_data)
        Y_smote.columns = ['stunting36']
        Y_smote = Y_smote.iloc[:, 0]
    else:
        X_smote = dep_var_data
        Y_smote = target_data
    return X_smote, Y_smote


def label_encoder(data):
    # Categorical boolean mask
    categorical_feature_mask = data.dtypes == object
    # filter categorical columns using mask and turn it into a list
    categorical_cols = data.columns[categorical_feature_mask].tolist()
    # instantiate labelencoder object
    le = LabelEncoder()
    # apply le on categorical feature columns
    data[categorical_cols] = data[categorical_cols].apply(lambda col: le.fit_transform(col))
    return data


def drop_data_from_column_list(data, list):
    return data.drop(list, axis=1)


def get_data_from_column_list(data, list):
    return data[list]


def smote(X, Y):
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, Y)
    X_res_df = pd.DataFrame(X_res, columns=X.columns)
    y_res_df = pd.DataFrame(y_res)
    y_res_df = y_res_df.iloc[:, :]
    return X_res_df, y_res_df


def one_hot_encoding(data):
    # Categorical boolean mask
    categorical_feature_mask = data.dtypes == object
    # filter categorical columns using mask and turn it into a list
    categorical_cols = data.columns[categorical_feature_mask].tolist()
    one_hot_encoded_data = pd.get_dummies(data, prefix_sep="__", columns=categorical_cols)
    one_hot_encoded_data.to_csv('one_hot_encoded_data.csv')
    return one_hot_encoded_data

warnings.filterwarnings("ignore")
main()
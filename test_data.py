import pandas as pd
import os
import ast
import random
import math
import numpy as np
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
from imblearn.over_sampling import SMOTE
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
# from skfeature.function.information_theoretical_based import CMIM, JMI, MIM, MRMR, RJMI, RelaxMRMR
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
# from skfeature.function.information_theoretical_based import CMIM, JMI, MIM, MRMR, RJMI, RelaxMRMR
import warnings
import time

# import matplotlib
# matplotlib.use('TkAgg')

def main():
    summary_data = pd.read_csv('summary_results.csv')
    test_model_list = ['train_feature_test', 'MH_feature_test', 'MH_feature_train', 'all_feature_train']
    for test_model in test_model_list:
        for index, data_row in summary_data.iterrows():
            imputaion = data_row.loc['imputaion']
            feature_selection_method = data_row.loc['feature selection method']
            is_smote = data_row.loc['smote']
            classifier_method = data_row.loc['classifier']
            train_file = ''
            test_file = 'non_imputed_full_test_data.csv'
            if imputaion == 'none':
                train_file = 'non_imputed_full_data.csv'
            if imputaion == 'wknn':
                train_file = 'weighted_knn_imputed_data.csv'
            if test_model == 'train_feature_test':
                feature_selected = ast.literal_eval(data_row.loc['features selected'])
                X_train, y_train = readdatafortest(train_file, feature_selected, is_smote, 'train')
                X_test, y_test = readdatafortest(test_file, feature_selected, is_smote, 'test')
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                auc, accuracy = run_classifier(X_train, y_train, X_test, y_test, classifier_method)
                print(classifier_method + ' - ' + feature_selection_method + ' : AUC - ' + str(auc) + ' , Accuracy - ' + str(accuracy))
                summary_data.loc[index, 'train feature test auc'] = auc
                summary_data.loc[index, 'train feature test accuracy'] = accuracy
            if test_model == 'MH_feature_test':
                feature_selected = ['fheight', 'mheight', 'infsex__male', 'finalgestdel', 'mweightgain', 'infweight_6month',
                                    'inflength_6month', 'infsex__female']
                X_train, y_train = readdatafortest(train_file, feature_selected, is_smote, 'train')
                X_test, y_test = readdatafortest(test_file, feature_selected, is_smote, 'test')
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                auc, accuracy = run_classifier(X_train, y_train, X_test, y_test, classifier_method)
                print(classifier_method + ' - ' + feature_selection_method + ' : AUC - ' + str(auc) + ' , Accuracy - ' + str(accuracy))
                summary_data.loc[index, 'MH feature test auc'] = auc
                summary_data.loc[index, 'MH feature test accuracy'] = accuracy
            if test_model == 'MH_feature_train':
                auc_list = []
                accuracy_list = []
                feature_selected = ['fheight', 'mheight', 'infsex__male', 'finalgestdel', 'mweightgain',
                                    'infweight_6month',
                                    'inflength_6month', 'infsex__female']
                X, Y = readdatafortest(train_file, feature_selected, is_smote, 'train')
                kf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=None)
                for train_index, test_index in kf.split(X):
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)
                    auc, accuracy = run_classifier(X_train, y_train, X_test, y_test, classifier_method)
                    auc_list.append(auc)
                    accuracy_list.append(accuracy)
                    auc_list = [auc for auc in auc_list if str(auc) != 'nan']
                print(classifier_method + ' - ' + feature_selection_method + ' : AUC - ' + str(np.mean(auc_list)) + ' , Accuracy - ' + str(np.mean(accuracy_list)))
                summary_data.loc[index, 'MH feature train auc'] = np.mean(auc_list)
                summary_data.loc[index, 'MH feature train auc std'] = np.std(auc_list)
                summary_data.loc[index, 'MH feature train accuracy'] = np.mean(accuracy_list)
                summary_data.loc[index, 'MH feature train accuracy std'] = np.std(accuracy_list)
            if test_model == 'all_feature_train':
                auc_list = []
                accuracy_list = []
                feature_selected = get_all_features(train_file, test_file)
                X, Y = readdatafortest(train_file, feature_selected, is_smote, 'train')
                kf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=None)
                for train_index, test_index in kf.split(X):
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)
                    auc, accuracy = run_classifier(X_train, y_train, X_test, y_test, classifier_method)
                    auc_list.append(auc)
                    accuracy_list.append(accuracy)
                    auc_list = [auc for auc in auc_list if str(auc) != 'nan']
                print(classifier_method + ' - ' + feature_selection_method + ' : AUC - ' + str(np.mean(auc_list)) + ' , Accuracy - ' + str(np.mean(accuracy_list)))
                summary_data.loc[index, 'All feature train auc'] = np.mean(auc_list)
                summary_data.loc[index, 'All feature train auc std'] = np.std(auc_list)
                summary_data.loc[index, 'All feature train accuracy'] = np.mean(accuracy_list)
                summary_data.loc[index, 'All feature train accuracy std'] = np.std(accuracy_list)
    summary_data.to_csv('test_data_summary_results.csv')


def get_all_features(train_file, test_file):
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    train_data_columns = train_data.columns.values
    test_data_columns = test_data.columns.values
    common_columns = list(set(train_data_columns).intersection(test_data_columns))
    data = get_data_from_column_list(train_data, common_columns)
    target_data_map = {'stunting36': {'yes': 1, 'no': 0}}
    data.replace(target_data_map, inplace=True)
    cols_to_remove = ['mage_cat', 'mheight_cat', 'mbmi_cat_enroll', 'fbmi_cat', 'preterm', 'mbmi_6month_cat',
                      'finalgestdel_cat', 'fedu_cat', 'pregnum_cat', 'fjob_cat', 'mjob_cat', 'medu_cat', ]
    for col in data.columns.values:
        if "36" in col:
            cols_to_remove.append(col)
        if "HAZ" in col:
            cols_to_remove.append(col)
        if "inflength" in col:
            cols_to_remove.append(col)
        if "stunting" in col:
            cols_to_remove.append(col)
    cols_to_remove.remove('inflength_6month')
    target_data = data.iloc[:, data.columns.get_loc('stunting36')]
    print(target_data)
    dep_var_data = drop_data_from_column_list(data, cols_to_remove)
    dep_var_data = one_hot_encoding(dep_var_data)
    print(dep_var_data.columns.values)
    return dep_var_data.columns.values


def run_classifier(X_train, y_train, X_test, y_test, classifier_method):
    if classifier_method == "MLP":
        classifier = MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=200)
    if classifier_method == "SVM":
        classifier = SVC(gamma='auto', probability=True)
    if classifier_method == "RDMF":
        classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=42)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    auc, accuracy = evaluation_metrics(y_test, y_pred)
    # print(list(zip(y_test,y_pred)))
    temp_dict = {"y_test": y_test, "y_ored": y_pred}
    dict_df = pd.DataFrame.from_dict(temp_dict)
    dict_df.to_csv(classifier_method+"test_cmp.csv")
    return auc, accuracy


def evaluation_metrics(y_test, y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=None)
    auc = metrics.auc(fpr, tpr)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    return auc, accuracy


def get_one_hot_encoded_data(data, feature_list):
    label_encoded_data = label_encoder(data)
    target_data = label_encoded_data.iloc[:, label_encoded_data.columns.get_loc('stunting36')]
    dep_var_data = get_data_from_column_list(data, feature_list)
    one_hot_data = one_hot_encoding(dep_var_data)
    return one_hot_data, target_data


def run_smote(X, Y):
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, Y)
    X_res_df = pd.DataFrame(X_res, columns=X.columns)
    y_res_df = pd.DataFrame(y_res)
    y_res_df = y_res_df.iloc[:, :]
    return X_res_df, y_res_df


def readdatafortest(filename, feature_selected, smote, datatype):
    data = pd.read_csv(filename)
    # yes_index = []
    # no_index = []
    # for index, data_item in data.iterrows():
    #     if data.loc[index, "stunting36"] == 'yes':
    #         yes_index.append(index)
    #     else:
    #         no_index.append(index)
    # print(yes_index)
    # print(no_index)
    # rand_no_index = random.choices(no_index, k=len(yes_index))
    # print(rand_no_index)
    # total_index = list(set(yes_index + rand_no_index))
    # print(total_index)
    # data = data.iloc[total_index, :]
    # print(data)
    datacopy = data.copy()
    target_data_map = {'stunting36': {'yes': 1, 'no': 0}}
    data.replace(target_data_map, inplace=True)
    target_data = data.iloc[:, data.columns.get_loc('stunting36')]
    replace_map = {'mjob': {1: 'farmer', 2: 'not working', 3: 'factory worker', 4: 'public services', 5: 'trader',
                            6: 'private services', 7: 'other'},
                   'medu': {1: '1', 2: 'incomplete primary', 3: 'complete primary', 4: 'complete secondary',
                            5: 'complete high', 6: 'vocational high school', 7: 'college', 8: '8'},
                   'fjob': {1: 'farmer', 2: 'not working', 3: 'factory worker', 4: 'public services', 5: 'trader',
                            6: 'private services', 7: 'other'},
                   'fedu': {1: '1', 2: 'incomplete primary', 3: 'complete primary', 4: 'complete secondary',
                            5: 'complete high', 6: 'vocational high school', 7: 'college', 8: '8'},
                   'pesticide': {1: 'yes', 0: 'no'},
                   'paint': {1: 'yes', 0: 'no'},
                   'grdfloor': {1: 'yes', 0: 'no'},
                   'highway': {1: 'yes', 0: 'no'},
                   'wellwater': {1: 'yes', 0: 'no'},
                   'mosqspay': {1: 'yes', 0: 'no'},
                   'industrial': {1: 'yes', 0: 'no'}
                   }
    if datatype == 'test':
        datacopy.replace(replace_map, inplace=True)
    one_hot_data = one_hot_encoding(datacopy)
    dep_var_data = get_data_from_column_list(one_hot_data, feature_selected)
    if smote == 'yes' and datatype == 'train':
        X_smote, Y_smote = run_smote(dep_var_data, target_data)
    else:
        X_smote = dep_var_data
        Y_smote = target_data
        X_smote = X_smote.iloc[:, :]  # independent columns
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


def one_hot_encoding(data):
    # Categorical boolean mask
    categorical_feature_mask = data.dtypes == object
    # filter categorical columns using mask and turn it into a list
    categorical_cols = data.columns[categorical_feature_mask].tolist()
    one_hot_encoded_data = pd.get_dummies(data, prefix_sep="__", columns=categorical_cols)
    # one_hot_encoded_data.to_csv('one_hot_encoded_data.csv')
    return one_hot_encoded_data

warnings.filterwarnings("ignore")
main()
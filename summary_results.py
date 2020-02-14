import pandas as pd
import random
import math
import os
import numpy as np
import scipy
from scipy.interpolate import splrep, splev
from itertools import combinations
from sklearn.neighbors.kde import KernelDensity
from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import GenericUnivariateSelect, chi2, f_classif, VarianceThreshold
from sklearn.feature_selection import SelectFpr
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import RepeatedKFold
from sklearn.neural_network import MLPClassifier
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
from operator import itemgetter
import csv
from sklearn.model_selection import RepeatedKFold
from scipy.ndimage.filters import gaussian_filter1d
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.svm import SVR
from scipy import stats
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
import matplotlib.pyplot as plt
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
from collections import Counter

def main():
    summary_data = pd.read_csv('summary_results.csv')
    feature_summary_data = pd.read_csv('top_features.csv')
    summary_data_columns = summary_data.columns.values
    feature_summary_data_columns = feature_summary_data.columns.values
    summary_dataframe = pd.DataFrame(columns=summary_data_columns)
    feature_summary_dataframe = pd.DataFrame(columns=feature_summary_data_columns)
    model_sum_dataframe, feat_sel_dataframe = get_model_summary_data('none', 'no', 'non_impute_feature_imp_results', summary_data_columns, feature_summary_data_columns)
    summary_dataframe = summary_dataframe.append(model_sum_dataframe, ignore_index=True)
    feature_summary_dataframe = feature_summary_dataframe.append(feat_sel_dataframe, ignore_index=True)
    model_sum_dataframe, feat_sel_dataframe = get_model_summary_data('none', 'yes', 'non_impute_feature_imp_smote_results', summary_data_columns, feature_summary_data_columns)
    summary_dataframe = summary_dataframe.append(model_sum_dataframe, ignore_index=True)
    feature_summary_dataframe = feature_summary_dataframe.append(feat_sel_dataframe, ignore_index=True)
    model_sum_dataframe, feat_sel_dataframe = get_model_summary_data('wknn', 'no', 'weighted_knn_feature_imp', summary_data_columns, feature_summary_data_columns)
    summary_dataframe = summary_dataframe.append(model_sum_dataframe, ignore_index=True)
    feature_summary_dataframe = feature_summary_dataframe.append(feat_sel_dataframe, ignore_index=True)
    model_sum_dataframe, feat_sel_dataframe = get_model_summary_data('wknn', 'yes', 'weighted_knn_smote_feature_imp', summary_data_columns, feature_summary_data_columns)
    summary_dataframe = summary_dataframe.append(model_sum_dataframe, ignore_index=True)
    feature_summary_dataframe = feature_summary_dataframe.append(feat_sel_dataframe, ignore_index=True)
    # print(summary_dataframe)
    summary_dataframe.to_csv('summary_results.csv', index=False)
    feature_summary_dataframe.to_csv('top_features.csv', index=False)


def get_model_summary_data(imputation, smote, model_directory, summary_data_columns, feature_summary_data_columns):
    path = os.getcwd()
    input_directory = path + '\\' + model_directory + '\\results\\'
    input_featureimp_directory = path + '\\' + model_directory + '\\results\\feature_importance\\'
    model_summary_dataframe = pd.DataFrame(columns=summary_data_columns)
    feature_selection_method_list = ['SelectFpr', 'SelectFdr', 'mutual_info_classif', 'feat_importance', 'L1_based',
                                     'selectkbest_f_classif', 'GenericUnivariateSelect', 'SelectPercentile',
                                     'VarianceThreshold']
    # feature_selection_method_list = ['SelectFpr']
    classifier_method_list = ['MLP', 'SVM', 'RDMF']
    # classifier_method_list = ['SVM']
    feat_sel_dataframe = pd.DataFrame(columns=feature_summary_data_columns)
    i = 1
    for feature_selection_method in feature_selection_method_list:
        feature_data = readfeatimpDataFromFile(feature_selection_method, input_featureimp_directory)
        top_15_feature = feature_data.head(15)
        top_15_feature_list = top_15_feature["feature"].tolist()
        temp_df = pd.DataFrame(columns=feature_summary_data_columns)
        temp_df.loc[i, 'imputation'] = imputation
        temp_df.loc[i, 'smote'] = smote
        temp_df.loc[i, 'feature selection method'] = feature_selection_method
        temp_df.loc[i, 'features identified'] = top_15_feature_list
        feat_sel_dataframe = feat_sel_dataframe.append(temp_df, ignore_index=True)

    classifier_auc_dict = {}
    for classifier_method in classifier_method_list:
        auc_accuracy_dataframe = pd.DataFrame(columns=summary_data_columns)
        class_mean_auc = 0
        class_auc_std = 1
        class_auc_list = []
        for feature_selection_method in feature_selection_method_list:
            # print(feature_selection_method)
            data = readDataFromFile(feature_selection_method, classifier_method, input_directory)
            data_measure, no_of_feature_list, auc_by_no_of_feat_list, accuracy_by_no_of_feat_list, auc_stdev_by_no_of_feat_list, accuracy_stdev_by_no_of_feat_list = addmeasurestoresult(
                data)
            top_auc_list, top_accuracy_list, top_feature_combination = top_auc_accuracy_list(data_measure)
            top_auc_list = [auc for auc in top_auc_list if str(auc) != 'nan']
            top_accuracy_list = [accuracy for accuracy in top_accuracy_list if str(accuracy) != 'nan']
            aucstdev = np.std(top_auc_list)
            auc_CI_value = (aucstdev * 1.96)/math.sqrt(50)
            accuracystdev = np.std(top_auc_list)
            accuracy_CI_value = (accuracystdev * 1.96) / math.sqrt(50)
            auc_mean = round(np.mean(top_auc_list), 2)
            if auc_mean > class_mean_auc:
                class_mean_auc = auc_mean
                class_auc_std = aucstdev
                class_auc_list = top_auc_list
            elif auc_mean == class_mean_auc:
                if aucstdev < class_auc_std:
                    class_mean_auc = auc_mean
                    class_auc_std = aucstdev
                    class_auc_list = top_auc_list
            auc_accuracy_dataframe.loc[feature_selection_method, "imputaion"] = imputation
            auc_accuracy_dataframe.loc[feature_selection_method, "smote"] = smote
            auc_accuracy_dataframe.loc[feature_selection_method, "classifier"] = classifier_method
            auc_accuracy_dataframe.loc[feature_selection_method, "feature selection method"] = feature_selection_method
            auc_accuracy_dataframe.loc[feature_selection_method, "mean AUC"] = round(np.mean(top_auc_list), 2)
            auc_accuracy_dataframe.loc[feature_selection_method, "stdev AUC"] = round(aucstdev,4)
            auc_accuracy_dataframe.loc[feature_selection_method, "CI AUC"] = str(round(np.mean(top_auc_list), 2)) +' +/_ '+ str(round(auc_CI_value, 3))
            auc_accuracy_dataframe.loc[feature_selection_method, "mean Accuracy"] = round(np.mean(top_accuracy_list), 2)
            auc_accuracy_dataframe.loc[feature_selection_method, "stdev Accuracy"] = round(accuracystdev, 4)
            auc_accuracy_dataframe.loc[feature_selection_method, "CI Accuracy"] = str(round(np.mean(top_accuracy_list), 2)) + ' +/- ' + str(round(accuracy_CI_value, 3))
            auc_accuracy_dataframe.loc[feature_selection_method, "features selected"] = top_feature_combination
        # print(auc_accuracy_dataframe)
        # auc_accuracy_dataframe.to_csv(classifier_method + 'temp.csv')
        sorted_auc_accuracy_dataframe = auc_accuracy_dataframe.sort_values(by=['mean AUC', 'stdev AUC', 'mean Accuracy',
                                                               'stdev Accuracy'], ascending=[False, True, False, True],
                                                           axis=0)

        # print(sorted_auc_accuracy_dataframe)
        # sorted_auc_accuracy_dataframe.to_csv(classifier_method + 'sort_temp.csv')
        # print(sorted_auc_accuracy_dataframe.head(1))
        # top_auc_accuracy_dataframe = sorted_auc_accuracy_dataframe
        top_auc_accuracy_dataframe = sorted_auc_accuracy_dataframe.head(1)
        model_summary_dataframe = model_summary_dataframe.append(top_auc_accuracy_dataframe, ignore_index=True)
        # print(summary_dataframe)
        classifier_auc_dict[classifier_method] = class_auc_list
    # print(model_summary_dataframe)
    model_summary_dataframe.to_csv(input_directory + 'model_summary_results.csv', index=False)
    F, p = stats.f_oneway(classifier_auc_dict['MLP'], classifier_auc_dict['SVM'], classifier_auc_dict['RDMF'])
    print(F)
    print(p)
    tucky_auc_list = []
    tucky_class_list = []
    for auc in classifier_auc_dict['MLP']:
        tucky_class_list.append('MLP')
        tucky_auc_list.append(auc)
    for auc in classifier_auc_dict['SVM']:
        tucky_class_list.append('SVM')
        tucky_auc_list.append(auc)
    for auc in classifier_auc_dict['RDMF']:
        tucky_class_list.append('RDMF')
        tucky_auc_list.append(auc)
    print(tucky_auc_list)
    print(tucky_class_list)
    mc = MultiComparison(tucky_auc_list, tucky_class_list)
    mc_results = mc.tukeyhsd()
    print(mc_results)
    return model_summary_dataframe, feat_sel_dataframe

def top_auc_accuracy_list(data_measure):
    data_measure_gt6_feature = data_measure.iloc[5:16,:]
    sorted_data_measure_dataframe = data_measure_gt6_feature.sort_values(by=['auc_mean', 'auc_stddev', 'accuracy_mean',
                                                                           'accuracy_stddev'],
                                                                       ascending=[False, True, False, True],
                                                                       axis=0)
    top_auc_datframe = sorted_data_measure_dataframe.head(1)
    # top_auc_datframe = data_measure_gt6_feature.nlargest(1, 'auc_mean')
    top_auc_list = []
    top_accuracy_list = []
    top_feature_combination = ''
    for index, top_result in top_auc_datframe.iterrows():
        # print(top_result.loc["feature combination"])
        top_feature_combination = top_result.loc["feature combination"]
        for i in range(1, 51):
            top_auc_list.append(top_result.loc["AUC-" + str(i)])
            top_accuracy_list.append(top_result.loc["ACCURACY-" + str(i)])
    return top_auc_list, top_accuracy_list, top_feature_combination

def addmeasurestoresult(data):
    index_list = []
    no_of_feature_list = []
    auc_by_no_of_feat_list = []
    accuracy_by_no_of_feat_list = []
    auc_stdev_by_no_of_feat_list = []
    accuracy_stdev_by_no_of_feat_list = []
    no_of_feature = 1
    no_of_feat_auc_list = {}
    tucky_index_list = []
    tucky_auc_list = []
    for index, data_row in data.iterrows():
        index_list.append(index)
        auc_list = []
        accuracy_list = []
        for i in range(1, 51):
            # print(feature_combination_result.loc["AUC-"+str(i)])
            auc_list.append(data_row.loc["AUC-" + str(i)])
            tucky_index_list.append(index)
            tucky_auc_list.append(data_row.loc["AUC-" + str(i)])
            accuracy_list.append(data_row.loc["ACCURACY-" + str(i)])
        auc_list = [auc for auc in auc_list if str(auc) != 'nan']
        accuracy_list = [accuracy for accuracy in accuracy_list if str(accuracy) != 'nan']
        no_of_feat_auc_list[index] = auc_list
        data.loc[index, "auc_stddev"] = round(np.std(auc_list),2)
        data.loc[index, "auc_mean"] = round(np.mean(auc_list),2)
        data.loc[index, "auc_min"] = np.min(auc_list)
        data.loc[index, "auc_max"] = np.max(auc_list)
        data.loc[index, "auc_median"] = np.median(auc_list)
        data.loc[index, "accuracy_stddev"] = round(np.std(accuracy_list),2)
        data.loc[index, "accuracy_mean"] = round(np.mean(accuracy_list),2)
        data.loc[index, "accuracy_min"] = np.min(accuracy_list)
        data.loc[index, "accuracy_max"] = np.max(accuracy_list)
        data.loc[index, "accuracy_median"] = np.median(accuracy_list)
        no_of_feature_list.append(no_of_feature)
        no_of_feature += 1
        accuracy_by_no_of_feat_list.append(np.mean(accuracy_list))
        auc_by_no_of_feat_list.append(np.mean(auc_list))
        auc_stdev_by_no_of_feat_list.append(np.std(auc_list))
        accuracy_stdev_by_no_of_feat_list.append(np.std(accuracy_list))
    # print(list(no_of_feat_auc_list.values()))
    # F, p = stats.f_oneway(no_of_feat_auc_list[0], no_of_feat_auc_list[1], no_of_feat_auc_list[2])
    # if p <= 0.05:
    #     mc = MultiComparison(tucky_auc_list, tucky_index_list)
    #     mc_results = mc.tukeyhsd()
    #     # print(mc_results)
    #     tucky_df = pd.DataFrame(data=mc_results._results_table.data[1:], columns=mc_results._results_table.data[0])
    #
    #     # print(tucky_df)
    #     tucky_true_df = tucky_df.loc[tucky_df['reject'] == True]
    #     if tucky_true_df.empty == False:
    #         print(tucky_df.loc[tucky_df['reject'] == True])
    # print(F)
    # print(p)
    # print(data)
    return data, no_of_feature_list, auc_by_no_of_feat_list, accuracy_by_no_of_feat_list, auc_stdev_by_no_of_feat_list, accuracy_stdev_by_no_of_feat_list

def readDataFromFile(feature_selection_method, classifier_method, input_directory):
    data = pd.read_csv(input_directory + feature_selection_method + "-" + classifier_method + ".csv")
    return data

def readfeatimpDataFromFile(feature_selection_method, input_directory):
    data = pd.read_csv(input_directory + feature_selection_method + "-importance.csv")
    return data


main()
import pandas as pd
import numpy as np
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
import matplotlib
matplotlib.use('TkAgg')
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
from skfeature.function.information_theoretical_based import CMIM, JMI, MIM, MRMR, RJMI, RelaxMRMR
import scipy
import seaborn as sns # data visualization library
from collections import Counter, OrderedDict
from operator import itemgetter

import time
print("-------REMEBER TO REMOVE DUMMY COLUMNS(id, unnamed) FROM DATASET AS EVERYTHING IS USED FOR PROCESS, SEARCH AND DELETE THE COLUMNS MANUALLY")

feature_selection_methods = ["all", "feature importance", "L1 based feature selection", "selectkbest",
                                 "GenericUnivariateSelect", "SelectPercentile", "selectFpr", "selectFdr", "mutual_info",
                                 "conditional_mutual_info_maximisation", "joint_mutual_info", "mutual_info_maximisation",
                                 "max_relevance_min_redundancy", "Recursive feature elimination",
                                 "Recursive feature elimination and cv selection"]

def generate_csv_from_dict(file_name, fieldnames, dataset):
    with open(file_name + '.csv', 'w', newline='') as csvfile:
        # fieldnames = ['first_name', 'last_name']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dataset)


def generate_result_csv_from_prediction(prediction, file_name="test_result"):
    field_names = ["ID_code", "target"]
    test_result = []
    i = 0
    for x in np.nditer(prediction):
        test_result.append({"ID_code": "test_"+str(i), "target": x})
        i += 1
    generate_csv_from_dict(dataset=test_result, fieldnames=field_names, file_name=file_name)



# def readDataFromFile():
#     # Creating Dataset and including the first row by setting no header as input
#
#     # Splitting the data into independent and dependent variables
#
#     data = pd.read_csv("santander-customer-transaction-prediction/train.csv")
#     test_dataset = pd.read_csv("santander-customer-transaction-prediction/test.csv")
#     X = data.iloc[:, 2:]  # independent columns
#     y = data.iloc[:, 1]
#     return X, y, test_dataset


def readDataFromFile():
    # Creating Dataset and including the first row by setting no header as input

    # Splitting the data into independent and dependent variables

    # data = pd.read_csv("child_growth_dataset/pre_processed_data.csv")
    # data = pd.read_csv("child_growth_dataset/knn_imputed_data.csv")
    # data = pd.read_csv("child_growth_dataset/knn_mode_imputed_data.csv")

    # data = pd.read_csv("child_growth_dataset/weighted_knn_imputed_data.csv")
    # data = pd.read_csv("child_growth_dataset/non_imputed_full_data.csv")
    data = pd.read_csv("child_growth_dataset/non_imputed_full_data_with_only_test_data_columns.csv")
    test_dataset = []
    y = data.iloc[:, data.columns.get_loc('stunting36')]
    data.drop(['stunting36'], axis=1, inplace=True)
    X = data.iloc[:, :]  # independent columns

    #Breast_Cancer_Wisconsin dataset read
    # data = pd.read_csv("breast_cancer_dataset/data.csv")
    # replace_map = {'diagnosis': {'M': 0, 'B': 1}}
    # data.replace(replace_map, inplace=True)
    # test_dataset = []
    # y = data.diagnosis  # M or B
    # list = ['Unnamed: 32', 'id', 'diagnosis']
    # X = data.drop(list, axis=1)


    return X, y, test_dataset


def evaluation_metrics(y_test, y_pred):
    mean_absolute_error = metrics.mean_absolute_error(y_test, y_pred)
    mean_squared_error = metrics.mean_squared_error(y_test, y_pred)
    root_mean_squared_error = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=None)
    tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    auc = metrics.auc(fpr, tpr)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    # print('Mean Absolute Error:', mean_absolute_error)
    # print('Mean Squared Error:', mean_squared_error)
    # print('Root Mean Squared Error:', root_mean_squared_error)

    print("accuracy_score:", accuracy)
    print("AUC:", auc)

    result = {"AUC": auc, "Accuracy": accuracy, "Mean Absolute Error": mean_absolute_error,
              "Mean Squared Error": mean_squared_error, "Root Mean Squared Error": root_mean_squared_error,
              "Sensitivity": sensitivity, "Specificity": specificity}
    return result

# def get_all_classifier_result():
#     # target column
#     # X_train, y_train, test_dataset = readDataFromFile()
#     # X_test = test_dataset.iloc[:, 1:].values
#     X_train, y_train, X_test = FeatureImportance()
#     print(X_train.head())
#     print(X_test.head())
#
#
#     # Feature Scaling
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train.values)
#     X_test = scaler.transform(X_test.values)
#
#     multi_layer_perceptron_classifier(X_train, y_train.values, X_test, test_dataset)


def results_plot():

    pickle_in = open("50_repeated_k_fold_results_without_RFE_KNN.pickle", "rb")
    results_dict = pickle.load(pickle_in)
    print(results_dict)
    # fig, axes = plt.subplots(len(results_dict.items()), 1)
    fig, axes = plt.subplots(3, 1)
    i = -1
    tempi = 0
    for key, value in results_dict.items():
        # if tempi == 4:
        #     break
        # else:
        #     i += 1
        #     axes[i].plot(100 * value["mlp_auc"], color='xkcd:cherry', marker='o', label='mlp')
        #     axes[i].plot(100 * value["svm_auc"], color='xkcd:royal blue', marker='o', label='svn')
        #     axes[i].plot(100 * value["rndf_auc"], color='xkcd:green', marker='o', label='Rndf')
        #     axes[i].set_xlabel('Repetition: ' + key)
        #     axes[i].set_ylabel('AUC (%)')
        #     axes[i].set_facecolor((1, 1, 1))
        #     axes[i].spines['left'].set_color('black')
        #     axes[i].spines['right'].set_color('black')
        #     axes[i].spines['top'].set_color('black')
        #     axes[i].spines['bottom'].set_color('black')
        #     axes[i].spines['left'].set_linewidth(1.5)
        #     axes[i].spines['right'].set_linewidth(0.5)
        #     axes[i].spines['top'].set_linewidth(0.5)
        #     axes[i].spines['bottom'].set_linewidth(0.5)
        #     axes[i].grid(linestyle='--', linewidth='0.5', color='grey', alpha=0.5)
        #     axes[i].legend(loc='best', ncol=2, mode=None, shadow=True, fancybox=True, fontsize='xx-small')

        if tempi <= 4 or tempi >= 8:
            tempi += 1
            continue
        else:
            i += 1
            print(i)

            axes[i].plot(100 * value["mlp_auc"], color='xkcd:cherry', marker='o', label='mlp')
            axes[i].plot(100 * value["svm_auc"], color='xkcd:royal blue', marker='o', label='svn')
            axes[i].plot(100 * value["rndf_auc"], color='xkcd:green', marker='o', label='Rndf')
            axes[i].set_xlabel('Repetition: ' + key)
            axes[i].set_ylabel('AUC (%)')
            axes[i].set_facecolor((1, 1, 1))
            axes[i].spines['left'].set_color('black')
            axes[i].spines['right'].set_color('black')
            axes[i].spines['top'].set_color('black')
            axes[i].spines['bottom'].set_color('black')
            axes[i].spines['left'].set_linewidth(1.5)
            axes[i].spines['right'].set_linewidth(0.5)
            axes[i].spines['top'].set_linewidth(0.5)
            axes[i].spines['bottom'].set_linewidth(0.5)
            axes[i].grid(linestyle='--', linewidth='0.5', color='grey', alpha=0.5)
            axes[i].legend(loc='best', ncol=2, mode=None, shadow=True, fancybox=True, fontsize='xx-small')

        tempi += 1


    plt.grid(True)
    plt.tight_layout()
    plt.show()


def temp_plot():
    mlp_auc = np.array([0.64299141, 0.71175523, 0.66634655, 0.7229064, 0.76204663, 0.68186546, 0.73208327, 0.7020202, 0.7108637, 0.76143273, 0.68936012, 0.76561378, 0.73863636, 0.69836538, 0.69331954, 0.69866071, 0.67955676, 0.74029126, 0.71255411, 0.70551286, 0.67793881, 0.68141034, 0.67588235, 0.80247158, 0.74568828, 0.68250631, 0.66287879, 0.79924242, 0.72561459, 0.69954913, 0.69669777, 0.72029412, 0.66552339, 0.78211876, 0.6695993, 0.70544383, 0.80059524, 0.71779412, 0.72770563, 0.71992424, 0.74829827, 0.74087252, 0.76469072, 0.67787648, 0.734375, 0.72749887, 0.80597015, 0.77800731, 0.66878103, 0.68232119])
    svm_auc = np.array([0.60526316, 0.62962963, 0.60810811, 0.64014778, 0.575, 0.58685853, 0.60797712, 0.56944444, 0.61842105, 0.64102564, 0.58333333, 0.62605887, 0.63888889, 0.61759615, 0.61764706, 0.58333333, 0.58593397, 0.64042996, 0.58066378, 0.58823529, 0.55555556, 0.56097561, 0.65926471, 0.63844832, 0.61290323, 0.65029437, 0.5530303, 0.63888889, 0.58974359, 0.640625, 0.55463029, 0.64705882, 0.58108108, 0.65533063, 0.58440767, 0.60228849, 0.61904762, 0.58823529, 0.57142857, 0.61871212, 0.609375, 0.63319926, 0.5875, 0.60603497, 0.57563744, 0.59090909, 0.65151515, 0.62903226, 0.52702703, 0.60526316])
    rndf_auc = np.array([0.72663802, 0.7173913, 0.68665112, 0.65188834, 0.725, 0.74757282, 0.7233434, 0.75631313, 0.88157895, 0.78456252, 0.71428571, 0.7, 0.83207071, 0.67759615, 0.66176471, 0.74739583, 0.66666667, 0.8165742, 0.76385281, 0.68866391, 0.81239936, 0.82408695, 0.70338235, 0.84615385, 0.78784733, 0.64785534, 0.66666667, 0.77272727, 0.66666667, 0.71875, 0.69497487, 0.70338235, 0.77870764, 0.72112011, 0.72970383, 0.79628988, 0.73809524, 0.69117647, 0.79747475, 0.6944697, 0.6200495, 0.6825495, 0.74484536, 0.71714608, 0.72691231, 0.67933062, 0.72478517, 0.69354839, 0.71111418, 0.78947368])

    fig, axes = plt.subplots(3, 1)

    axes[0].plot(100 * mlp_auc, color='xkcd:cherry', marker='o', label='mlp')
    axes[0].plot(100 * svm_auc, color='xkcd:royal blue', marker='o', label='svn')
    axes[0].plot(100 * rndf_auc, color='xkcd:green', marker='o', label='rndf')
    axes[0].set_xlabel('Repetition')
    axes[0].set_ylabel('AUC (%)')
    axes[0].set_facecolor((1, 1, 1))
    axes[0].spines['left'].set_color('black')
    axes[0].spines['right'].set_color('black')
    axes[0].spines['top'].set_color('black')
    axes[0].spines['bottom'].set_color('black')
    axes[0].spines['left'].set_linewidth(0.5)
    axes[0].spines['right'].set_linewidth(0.5)
    axes[0].spines['top'].set_linewidth(0.5)
    axes[0].spines['bottom'].set_linewidth(0.5)
    axes[0].grid(linestyle='--', linewidth='0.5', color='grey', alpha=0.5)
    axes[0].legend(loc='best', ncol=2, mode=None, shadow=True, fancybox=True, fontsize='xx-small')

    plt.grid(True)
    plt.tight_layout()
    plt.show()



def line_graph():
    X, y, test_dataset = readDataFromFile()

    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.25, random_state=21)


    pickle_in = open("50_repeated_k_fold_results_knn_MI.pickle", "rb")
    results_dict = pickle.load(pickle_in)
    print("results_dict:",results_dict)
    for key, value in results_dict.items():
        if key == "all":
            continue

        feat_list = value['feat_list']
        x_feat_len = [[]] * 3
        y_auc = [[]] * 3
        prev_mlp_auc = 0.0
        prev_svm_auc = 0.0
        prev_rndf_auc = 0.0
        mlp_auc = []
        svm_auc = []
        rndf_auc = []


        for i in range(len(feat_list)):
            slic_feat_list = feat_list[:(i+1)]
            # print("====================")
            # print(slic_feat_list)
            # print(X_train)
            cols_index = [X.columns.get_loc(col) for col in slic_feat_list]
            # print(cols_index)
            X_feat = X_train[:, cols_index]
            y_feat = y_train
            X_feat_test = X_test[:, cols_index]
            y_feat_test = y_test

            features_imp_result = get_model_results(X_feat, y_feat, X_feat_test, y_feat_test, slic_feat_list, key)
            x_feat_len[0] = x_feat_len[0] + [len(slic_feat_list)]
            x_feat_len[1] = x_feat_len[1] + [len(slic_feat_list)]
            x_feat_len[2] = x_feat_len[2] + [len(slic_feat_list)]

            y_auc[0] = y_auc[0] + [features_imp_result[0]["AUC"]]
            y_auc[1] = y_auc[1] + [features_imp_result[1]["AUC"]]
            y_auc[2] = y_auc[2] + [features_imp_result[2]["AUC"]]

            if features_imp_result[0]["AUC"] > prev_mlp_auc:
                prev_mlp_auc = features_imp_result[0]["AUC"]
                mlp_auc = [set(slic_feat_list), prev_mlp_auc]

            if features_imp_result[1]["AUC"] > prev_svm_auc:
                prev_svm_auc = features_imp_result[1]["AUC"]
                svm_auc = [set(slic_feat_list), prev_svm_auc]

            if features_imp_result[2]["AUC"] > prev_rndf_auc:
                prev_rndf_auc = features_imp_result[2]["AUC"]
                rndf_auc = [set(slic_feat_list), prev_rndf_auc]



        # plt.set_xlabel('Feature length: ' + key)
        # plt.set_ylabel('AUC (%)')
        print("-------------------", key, "-------------------")
        print(mlp_auc)
        print(svm_auc)
        print(rndf_auc)
        mlp_txt = "MLP: " + ", ".join(mlp_auc[0]) + " - " + str(round(mlp_auc[1], 4))
        svm_txt = "SVM: " + ", ".join(svm_auc[0]) + " - " + str(round(svm_auc[1], 4))
        rndf_txt = "RNDF: " + ", ".join(rndf_auc[0]) + " - " + str(round(rndf_auc[1], 4))
        plt.text(0.12, 0.13, mlp_txt, fontsize=10, transform=plt.gcf().transFigure)
        plt.text(0.12, 0.09, svm_txt, fontsize=10, transform=plt.gcf().transFigure)
        plt.text(0.12, 0.05, rndf_txt, fontsize=10, transform=plt.gcf().transFigure)
        plt.suptitle(key)
        plt.plot(x_feat_len[0], y_auc[0], 'co-', label='MLP')
        plt.plot(x_feat_len[1], y_auc[1], 'bo-', label='SVM')
        plt.plot(x_feat_len[2], y_auc[2], 'go-', label='RNDF')
        plt.subplots_adjust(bottom=0.25)
        plt.legend()
        plt.show()



def perform_repeated_k_fold_on_model(n_splits, n_repeats, read_pickle_filename, write_pickle_filename, part_pickle_filename):
    start = time.time()
    X, y, test_dataset = readDataFromFile()
    # n_splits = 5
    # n_repeats = 5
    kf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=None)
    # i = 0

    classif_pos = {"mlp": 0, "svm": 1, "rndf": 2}
    new_results_dict = {}
    classifiers_str = ["mlp", "svm", "rndf"]
    metrics_str = ["AUC", "Accuracy", "Sensitivity", "Specificity"]
    metrics_eval = list(map(lambda x: x[0] + "_" + x[1], list(itertools.product(classifiers_str, metrics_str))))
    for fsm in feature_selection_methods:
        temp_metrics_dict = {}
        for metric in metrics_eval:
            temp_metrics_dict[metric] = {}
        new_results_dict[fsm] = temp_metrics_dict


    # new_results_dict = {"feature importance": {"mlp_auc": {}, "svm_auc": {}, "rndf_auc": {}},
    #                 "L1 based feature selection": {"mlp_auc": {}, "svm_auc": {}, "rndf_auc": {}},
    #                 "selectkbest": {"mlp_auc": {}, "svm_auc": {}, "rndf_auc": {}},
    #                 "GenericUnivariateSelect": {"mlp_auc": {}, "svm_auc": {}, "rndf_auc": {}},
    #                 "SelectPercentile": {"mlp_auc": {}, "svm_auc": {}, "rndf_auc": {}},
    #                 "selectFpr": {"mlp_auc": {}, "svm_auc": {}, "rndf_auc": {}},
    #                 "selectFdr": {"mlp_auc": {}, "svm_auc": {}, "rndf_auc": {}},
    #                 "mutual_info": {"mlp_auc": {}, "svm_auc": {}, "rndf_auc": {}},
    #                 "Recursive feature elimination": {"mlp_auc": {}, "svm_auc": {}, "rndf_auc": {}},
    #                 "Recursive feature elimination and cv selection": {"mlp_auc": {}, "svm_auc": {}, "rndf_auc": {}}}



    # pickle_in = open("50_repeated_k_fold_results_knn_mode.pickle", "rb")
    pickle_in = open(read_pickle_filename, "rb")
    results_dict = pickle.load(pickle_in)
    # print(results_dict)
    for key, value in results_dict.items():
        if key == "all" or 'feat_list' not in value:
            continue
        value['feat_list'].sort(key=lambda item:item[1], reverse=True)
        # print("KEY:",key)
        # print("features:", value['feat_list'])

    first_iter = True
    k_fold_iter = 0
    for train_index, test_index in kf.split(X):
        # print("Train:", train_index, "Validation:", test_index)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # pickle_in = open("50_repeated_k_fold_results_knn_mode.pickle", "rb")
        # results_dict = pickle.load(pickle_in)
        # print("results_dict:", results_dict)

        for key, value in results_dict.items():
            if key == "all" or 'feat_list' not in value:
                continue

            feat_list = value['feat_list']

            # print(feat_list)

            # feat_list.sort(key=lambda item:item[1], reverse=True)
            # print()
            feat_list = [feat[0] for feat in feat_list[:10]]
            # print(feat_list)


            # for comb_len in range(1,len(feat_list)+1):
            for comb_len in range(6,len(feat_list)+1):
                feat_combs = list(itertools.combinations(feat_list, comb_len))

                for feat_comb in feat_combs:
                    slic_feat_list = list(feat_comb)
                    X_feat = X_train[slic_feat_list]
                    y_feat = y_train
                    X_feat_test = X_test[slic_feat_list]
                    y_feat_test = y_test

                    features_imp_result = get_model_results(X_feat, y_feat, X_feat_test, y_feat_test, slic_feat_list, key)

                    feat_key = ", ".join(slic_feat_list)

                    if first_iter:
                        for metric in metrics_eval:
                            new_results_dict[key][metric][feat_key] = []


                    for metric in metrics_eval:
                        mt_split = metric.split("_")
                        classif = mt_split[0]
                        eval_metric = mt_split[1]

                        new_results_dict[key][metric][feat_key] += [features_imp_result[classif_pos[classif]][eval_metric]]


        first_iter = False
        k_fold_iter += 1
        if k_fold_iter == 10 or k_fold_iter == 2 or k_fold_iter == 5 or k_fold_iter == 8 or k_fold_iter == 1:
            pickle_out = open(str(k_fold_iter) + part_pickle_filename, "wb")
            pickle.dump(new_results_dict, pickle_out)
            pickle_out.close()
            print("perform_repeated_k_fold_on_model. time taken: ", time.strftime("%H:%M:%S", time.gmtime(time.time() - start)))
            # Write-Overwrites
            file1 = open("time_perform_repeated_k_fold_on_model.txt", "a")  # write mode
            file1.write("\n" + str(k_fold_iter) + "perform_repeated_k_fold_on_model. time taken: " + time.strftime("%H:%M:%S", time.gmtime(time.time() - start)))
            file1.close()

    # pickle_out = open(str(n_splits*n_repeats)+"_model_cumulative_k_fold_results_knn.pickle", "wb")
    pickle_out = open(write_pickle_filename, "wb")
    pickle.dump(new_results_dict, pickle_out)
    pickle_out.close()


def line_graph_model(pickle_file_name, is_accuracy=False):
    # pickle_in = open("10_model_cumulative_k_fold_results_knn.pickle", "rb")
    pickle_in = open(pickle_file_name, "rb")
    results_dict = pickle.load(pickle_in)
    classifiers_str = ["mlp", "svm", "rndf"]
    # print(results_dict)
    for key, model_value in results_dict.items():
        if key == "all":
            continue

        x_feat_len = [[]] * 3
        y_auc = [[]] * 3
        # prev_mlp_auc = 0.0
        # prev_svm_auc = 0.0
        # prev_rndf_auc = 0.0
        all_classif_auc = [[]] * 3
        # mlp_auc = [[]] * 3
        # svm_auc = []
        # rndf_auc = []

        # print("key:", key, " value:", model_value["mlp_AUC"])

        eval_metric = "_AUC"
        corres_eval_metric = "_Accuracy"
        figure_title = " - plotting AUC"
        plt.ylabel("AUC")

        if is_accuracy:
            eval_metric = "_Accuracy"
            corres_eval_metric = "_AUC"
            figure_title = " - plotting ACCURACY"
            plt.ylabel("Accuracy")


        for classif_str in classifiers_str:
            temp_auc_mean = {}
            temp_auc_mean_corres_acc_index = {}
            temp_auc_mean_feat_list = {}
            for k, v in model_value[classif_str + eval_metric].items():
                temp_auc_mean_key = len(k.split(", "))
                # print(classif_str+"_AUC", " key:", k, " val:", v)
                if temp_auc_mean_key not in temp_auc_mean:
                    temp_auc_mean[temp_auc_mean_key] = []
                    temp_auc_mean_corres_acc_index[temp_auc_mean_key] = []
                    temp_auc_mean_feat_list[temp_auc_mean_key] = []
                # max_value = max(v)
                mean_value = statistics.mean(v)
                # print("mean_value:",mean_value)

                # max_index = v.index(max_value)
                # print("max_index:", max_index)
                temp_auc_mean[temp_auc_mean_key].append(mean_value)
                # print("model_value["+classif_str+"_Accuracy][k]:", model_value[classif_str+"_Accuracy"][k])
                temp_auc_mean_corres_acc_index[temp_auc_mean_key].append(statistics.mean(model_value[classif_str + corres_eval_metric][k]))
                temp_auc_mean_feat_list[temp_auc_mean_key].append(k)
                # print("value:", model_value[classif_str+corres_eval_metric][k])
                # print("value:", k)



            # print("temp_auc_max:", temp_auc_mean)
            # print("temp_auc_max_corres_acc_index:", temp_auc_mean_corres_acc_index)
            # print("temp_auc_max_feat_list:", temp_auc_mean_feat_list)
            prev_auc = 0.0
            for k,v in temp_auc_mean.items():
                max_value = max(v)
                max_index = v.index(max_value)
                classif_index = classifiers_str.index(classif_str)
                # x_feat_len[classif_index].append(k)
                x_feat_len[classif_index] = x_feat_len[classif_index] + [k]
                # y_auc[classif_index].append(max_value)
                y_auc[classif_index] = y_auc[classif_index] + [max_value]

                if max_value > prev_auc:
                    prev_auc = max_value
                    all_classif_auc[classif_index] = [temp_auc_mean_feat_list[k][max_index], prev_auc, temp_auc_mean_corres_acc_index[k][max_index]]

                # if features_imp_result[1]["AUC"] > prev_svm_auc:
                #     prev_svm_auc = features_imp_result[1]["AUC"]
                #     svm_auc = [set(slic_feat_list), prev_svm_auc]
                #
                # if features_imp_result[2]["AUC"] > prev_rndf_auc:
                #     prev_rndf_auc = features_imp_result[2]["AUC"]
                #     rndf_auc = [set(slic_feat_list), prev_rndf_auc]
                #
                # corres_acc = temp_auc_max_corres_acc_index[k][max_index]
                # feat_list = temp_auc_max_feat_list[k][max_index]

        if x_feat_len == [[]] *3:
            continue

        sort_temp = [sorted(list(zip(x_feat_len[i],y_auc[i])), key=lambda x:x[0]) for i in range(len(x_feat_len))]
        sort_temp = [list(zip(*sort_temp[i])) for i in range(len(x_feat_len))]
        x_feat_len = [list(ll[0]) for ll in sort_temp]
        y_auc = [list(ll[1]) for ll in sort_temp]
        print("-------------------", key, "-------------------")
        print("x_feat_len:", x_feat_len)
        print("y_auc:", y_auc)
        # print(mlp_auc)
        # print(svm_auc)
        # print(rndf_auc)
        mlp_txt = "MLP: " + all_classif_auc[0][0] + " - " + str(round(all_classif_auc[0][1], 4))  + " - " + str(round(all_classif_auc[0][2], 4))
        svm_txt = "SVM: " + all_classif_auc[1][0] + " - " + str(round(all_classif_auc[1][1], 4))  + " - " + str(round(all_classif_auc[1][2], 4))
        rndf_txt = "RNDF: " + all_classif_auc[2][0] + " - " + str(round(all_classif_auc[2][1], 4))  + " - " + str(round(all_classif_auc[2][2], 4))
        # svm_txt = "SVM: " + ", ".join(svm_auc[0]) + " - " + str(round(svm_auc[1], 4))
        # rndf_txt = "RNDF: " + ", ".join(rndf_auc[0]) + " - " + str(round(rndf_auc[1], 4))
        plt.text(0.12, 0.13, mlp_txt, fontsize=10, transform=plt.gcf().transFigure)
        plt.text(0.12, 0.09, svm_txt, fontsize=10, transform=plt.gcf().transFigure)
        plt.text(0.12, 0.05, rndf_txt, fontsize=10, transform=plt.gcf().transFigure)
        plt.suptitle(key + figure_title)
        plt.plot(x_feat_len[0], y_auc[0], 'co-', label='MLP')
        plt.plot(x_feat_len[1], y_auc[1], 'bo-', label='SVM')
        plt.plot(x_feat_len[2], y_auc[2], 'go-', label='RNDF')
        plt.subplots_adjust(bottom=0.25)
        plt.xlabel("# of features")
        plt.legend()
        plt.show()
        # return 0


            # model_value[classif_str+"_Accuracy"]

        # feat_list = model_value['feat_list']

        # prev_mlp_auc = 0.0
        # prev_svm_auc = 0.0
        # prev_rndf_auc = 0.0
        # mlp_auc = []
        # svm_auc = []
        # rndf_auc = []


        # for ky, each_feat_auc in model_value.items():
        #     for k, feat_auc in each_feat_auc.items():
        #         min_auc = min(feat_auc)
        #         max_auc = max(feat_auc)
        #         mean_auc = statistics.mean(feat_auc)
        #         median_auc = statistics.median(feat_auc)







        # plt.set_xlabel('Feature length: ' + key)
        # plt.set_ylabel('AUC (%)')



def updated_dict_with_model_results(key, results_dict, i, all_features_result, metrics_eval):
    classif_pos = {"mlp": 0, "svm": 1, "rndf": 2}
    for metric in metrics_eval:
        mt_split = metric.split("_")
        classif = mt_split[0]
        eval_metric = mt_split[1]

        results_dict[key][metric][i] = all_features_result[classif_pos[classif]][eval_metric]
    return results_dict


def perform_repeated_k_fold(n_splits, n_repeats, pickle_filename):
    start = time.time()
    X, y, test_dataset = readDataFromFile()
    # n_splits = 5
    # n_repeats = 10
    num_of_fold = n_splits*n_repeats
    kf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=None)

    results_dict = {}
    classifiers_str = ["mlp", "svm", "rndf"]
    metrics_str = ["AUC", "Accuracy", "Sensitivity", "Specificity"]
    metrics_eval = list(map(lambda x:x[0]+"_"+x[1], list(itertools.product(classifiers_str, metrics_str))))
    for fsm in feature_selection_methods:
        temp_metrics_dict = {}
        for metric in metrics_eval:
            # temp_metrics_dict[metric] = np.zeros(n_splits * n_repeats)
            temp_metrics_dict[metric] = [0.0 for _ in range(n_splits * n_repeats)]
        results_dict[fsm] = temp_metrics_dict

    # print(results_dict)

    i = 0
    for train_index, test_index in kf.split(X):
        # print("Train:", len(train_index), "Validation:", len(test_index))
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        all_features_result = get_model_results(X_train, y_train, X_test, y_test, None, "all")
        results_dict = updated_dict_with_model_results("all", results_dict, i, all_features_result, metrics_eval)
        results_dict["all"]["feat_list"] = "all"

        X_raw = X_train
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        # X_test = scaler.transform(X_test)


        print("------------------feature importance result----------------")
        fs_method_name = "feature importance"
        results_dict = feature_selection_result_update(num_of_fold, i, fs_method_name, FeatureImportance, X_raw, X_train, y_train, X_test, y_test, metrics_eval, results_dict)


        print("------------------L1_based_feature_selection result----------------")
        fs_method_name = "L1 based feature selection"
        results_dict = feature_selection_result_update(num_of_fold, i, fs_method_name, L1_based_feature_selection, X_raw, X_train, y_train, X_test, y_test, metrics_eval, results_dict)


        print("------------------feature_selectkbest_f_classif result----------------")
        fs_method_name = "selectkbest"
        results_dict = feature_selection_result_update(num_of_fold, i, fs_method_name, feature_selectkbest_f_classif,
                                                       X_raw, X_train, y_train, X_test, y_test, metrics_eval,
                                                       results_dict)

        print("------------------feature_GenericUnivariateSelect result----------------")
        fs_method_name = "GenericUnivariateSelect"
        results_dict = feature_selection_result_update(num_of_fold, i, fs_method_name, feature_GenericUnivariateSelect,
                                                       X_raw, X_train, y_train, X_test, y_test, metrics_eval,
                                                       results_dict)

        print("------------------feature_SelectPercentile result----------------")
        fs_method_name = "SelectPercentile"
        results_dict = feature_selection_result_update(num_of_fold, i, fs_method_name, feature_SelectPercentile,
                                                       X_raw, X_train, y_train, X_test, y_test, metrics_eval,
                                                       results_dict)

        print("------------------feature_SelectFpr result----------------")
        fs_method_name = "selectFpr"
        results_dict = feature_selection_result_update(num_of_fold, i, fs_method_name, feature_SelectFpr,
                                                       X_raw, X_train, y_train, X_test, y_test, metrics_eval,
                                                       results_dict)


        print("------------------feature_SelectFdr result----------------")
        fs_method_name = "selectFdr"
        results_dict = feature_selection_result_update(num_of_fold, i, fs_method_name, feature_SelectFdr,
                                                       X_raw, X_train, y_train, X_test, y_test, metrics_eval,
                                                       results_dict)


        print("------------------feature_mutual_info result----------------")
        fs_method_name = "mutual_info"
        results_dict = feature_selection_result_update(num_of_fold, i, fs_method_name, feature_mutual_info,
                                                       X_raw, X_train, y_train, X_test, y_test, metrics_eval,
                                                       results_dict)


        print("------------------feature_conditional_mutual_info_maximisation result----------------")
        fs_method_name = "conditional_mutual_info_maximisation"
        results_dict = feature_selection_result_update(num_of_fold, i, fs_method_name, feature_conditional_mutual_info_maximisation,
                                                       X_raw, X_train, y_train, X_test, y_test, metrics_eval,
                                                       results_dict)


        print("------------------feature_joint_mutual_info result----------------")
        fs_method_name = "joint_mutual_info"
        results_dict = feature_selection_result_update(num_of_fold, i, fs_method_name, feature_joint_mutual_info,
                                                       X_raw, X_train, y_train, X_test, y_test, metrics_eval,
                                                       results_dict)


        print("------------------feature_mutual_info_maximisation result----------------")
        fs_method_name = "mutual_info_maximisation"
        results_dict = feature_selection_result_update(num_of_fold, i, fs_method_name, feature_mutual_info_maximisation,
                                                       X_raw, X_train, y_train, X_test, y_test, metrics_eval,
                                                       results_dict)

        print("------------------feature_max_relevance_min_redundancy result----------------")
        fs_method_name = "max_relevance_min_redundancy"
        results_dict = feature_selection_result_update(num_of_fold, i, fs_method_name, feature_max_relevance_min_redundancy,
                                                       X_raw, X_train, y_train, X_test, y_test, metrics_eval,
                                                       results_dict)



        # print("------------------feature_RFE result----------------")
        # fs_method = "Recursive feature elimination"
        # X_feat, y_feat, temp, feat_list = feature_RFE(X_train, y_train)
        # RFE_result = get_model_results(X_feat, y_feat, X_test, y_test, feat_list, fs_method)
        # results_dict[fs_method]["mlp_auc"][i] = RFE_result[0]['AUC']
        # results_dict[fs_method]["svm_auc"][i] = RFE_result[1]['AUC']
        # results_dict[fs_method]["rndf_auc"][i] = RFE_result[2]['AUC']
        # if "feat_list" not in results_dict[fs_method]:
        #     results_dict[fs_method]["feat_list"] = feat_list.split(", ")
        # results_dict[fs_method]["feat_list"] = list(set(feat_list.split(", ")).intersection(set(results_dict[fs_method]["feat_list"])))

        #
        # print("------------------feature_RFECV result----------------")
        # fs_method = "Recursive feature elimination and cross-validated selection"
        # X_feat, y_feat, temp, feat_list = feature_RFECV(X_train, y_train)
        # RFECV_result = get_model_results(X_feat, y_feat, X_test, y_test, feat_list, fs_method)
        # results_dict["Recursive feature elimination and cv selection"]["mlp_auc"][i] = RFECV_result[0]['AUC']
        # results_dict["Recursive feature elimination and cv selection"]["svm_auc"][i] = RFECV_result[1]['AUC']
        # results_dict["Recursive feature elimination and cv selection"]["rndf_auc"][i] = RFECV_result[2]['AUC']
        # if "feat_list" not in results_dict["Recursive feature elimination and cv selection"]:
        #     results_dict["Recursive feature elimination and cv selection"]["feat_list"] = feat_list.split(", ")
        # results_dict["Recursive feature elimination and cv selection"]["feat_list"] = list(set(feat_list.split(", ")).intersection(set(results_dict["Recursive feature elimination and cv selection"]["feat_list"])))


        i += 1

    # pickle_out = open(str(n_splits*n_repeats)+"_repeated_k_fold_results_knn_mode.pickle", "wb")
    pickle_out = open(pickle_filename, "wb")
    pickle.dump(results_dict, pickle_out)
    pickle_out.close()
    file1 = open("time_perform_repeated_k_fold_on_model.txt", "a")  # write mode
    file1.write("\n" + str(n_splits*n_repeats) + "_perform_repeated_k_fold. time taken: " + time.strftime("%H:%M:%S",
                                                                                                           time.gmtime(
                                                                                                               time.time() - start)))
    file1.close()
    print("perform_repeated_k_fold. time taken: ", time.strftime("%H:%M:%S", time.gmtime(time.time() - start)))


    # fig, axes = plt.subplots(len(results_dict.items()), 1)
    # i = 0
    # for key, value in results_dict.items():
    #
    #     axes[i].plot(100 * value["mlp_AUC"], color='xkcd:cherry', marker='o', label='mlp')
    #     axes[i].plot(100 * value["svm_AUC"], color='xkcd:royal blue', marker='o', label='svn')
    #     axes[i].plot(100 * value["rndf_AUC"], color='xkcd:green', marker='o', label='rndf')
    #     axes[i].set_xlabel('Repetition: '+key)
    #     axes[i].set_ylabel('AUC (%)')
    #     axes[i].set_facecolor((1, 1, 1))
    #     axes[i].spines['left'].set_color('black')
    #     axes[i].spines['right'].set_color('black')
    #     axes[i].spines['top'].set_color('black')
    #     axes[i].spines['bottom'].set_color('black')
    #     axes[i].spines['left'].set_linewidth(1.5)
    #     axes[i].spines['right'].set_linewidth(0.5)
    #     axes[i].spines['top'].set_linewidth(0.5)
    #     axes[i].spines['bottom'].set_linewidth(0.5)
    #     axes[i].grid(linestyle='--', linewidth='0.5', color='grey', alpha=0.5)
    #     axes[i].legend(loc='best', ncol=2, mode=None, shadow=True, fancybox=True, fontsize='xx-small')
    #
    #     i += 1

    # axes[1].plot(100 * mlp_auc, color='xkcd:royal blue', marker='o')
    # axes[1].set_xlabel('Repetition')
    # axes[1].set_ylabel('AUC(%)')
    # axes[1].set_facecolor((1, 1, 1))
    # axes[1].spines['left'].set_color('black')
    # axes[1].spines['right'].set_color('black')
    # axes[1].spines['top'].set_color('black')
    # axes[1].spines['bottom'].set_color('black')
    # axes[1].spines['left'].set_linewidth(0.5)
    # axes[1].spines['right'].set_linewidth(0.5)
    # axes[1].spines['top'].set_linewidth(0.5)
    # axes[1].spines['bottom'].set_linewidth(0.5)
    # axes[1].grid(linestyle='--', linewidth='0.5', color='grey', alpha=0.5)
    #
    # axes[2].plot(100 * mlp_auc, color='xkcd:emerald', marker='o')
    # axes[2].set_xlabel('Repetition')
    # axes[2].set_ylabel('AUC (%)')
    # axes[2].set_facecolor((1, 1, 1))
    # axes[2].spines['left'].set_color('black')
    # axes[2].spines['right'].set_color('black')
    # axes[2].spines['top'].set_color('black')
    # axes[2].spines['bottom'].set_color('black')
    # axes[2].spines['left'].set_linewidth(0.5)
    # axes[2].spines['right'].set_linewidth(0.5)
    # axes[2].spines['top'].set_linewidth(0.5)
    # axes[2].spines['bottom'].set_linewidth(0.5)
    # axes[2].grid(linestyle='--', linewidth='0.5', color='grey', alpha=0.5)

    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

def feature_selection_result_update(num_of_fold, i, fs_method_name, fs_method, X_raw, X_train, y_train, X_test, y_test, metrics_eval, results_dict):
    X_feat, y_feat, temp, feat_list, feat_list_with_imp = fs_method(X_train, X_raw, y_train)
    if feat_list == [] or feat_list == "":
        return results_dict
    features_imp_result = get_model_results(X_feat, y_feat, X_test, y_test, feat_list, fs_method_name)

    results_dict = updated_dict_with_model_results(fs_method_name, results_dict, i, features_imp_result, metrics_eval)

    if "feat_list" not in results_dict[fs_method_name]:
        # results_dict[fs_method_name]["feat_list"] = feat_list.split(", ")
        results_dict[fs_method_name]["feat_list"] = feat_list_with_imp
        feat_list_with_imp = []
    results_dict[fs_method_name]["feat_list"] = merge_feat_imp_list(results_dict[fs_method_name]["feat_list"], feat_list_with_imp,
                                                               i, num_of_fold)

    return results_dict


def feat_grt_5_top_5_mean(pickle_file_name, is_accuracy=False):
    start = time.time()
    # pickle_in = open("10_model_cumulative_k_fold_results_knn.pickle", "rb")
    pickle_in = open(pickle_file_name, "rb")
    results_dict = pickle.load(pickle_in)
    classifiers_str = ["mlp", "svm", "rndf"]
    # print(results_dict)
    top_5_mean_50_fold_classif_result = {}


    del results_dict["Recursive feature elimination"]
    del results_dict["Recursive feature elimination and cv selection"]
    for key, model_value in results_dict.items():
        if key == "all":
            continue
        print()
        print("****************", key, "*********************")
        x_feat_len = [[]] * 3
        y_auc = [[]] * 3
        all_classif_auc = {}

        eval_metric = "_AUC"
        corres_eval_metric = "_Accuracy"
        figure_title = " - plotting AUC"

        if is_accuracy:
            eval_metric = "_Accuracy"
            corres_eval_metric = "_AUC"
            figure_title = " - plotting ACCURACY"

        top_5_mean_50_fold_classif_result[key] = {}
        for classif_str in classifiers_str:

            # if classif_str + eval_metric not in top_5_mean_50_fold_classif_result[key]:
            top_5_mean_50_fold_classif_result[key][classif_str + eval_metric] = {}

            temp_auc_mean = {}
            temp_auc_mean_corres_acc_index = {}
            temp_auc_mean_feat_list = {}
            for k, v in model_value[classif_str + eval_metric].items():
                temp_auc_mean_key = len(k.split(", "))
                if temp_auc_mean_key < 6:
                    continue
                # print(classif_str+"_AUC", " key:", k, " val:", v)
                if temp_auc_mean_key not in temp_auc_mean:
                    temp_auc_mean[temp_auc_mean_key] = []
                    temp_auc_mean_corres_acc_index[temp_auc_mean_key] = []
                    temp_auc_mean_feat_list[temp_auc_mean_key] = []
                # max_value = max(v)
                mean_value = statistics.mean(v)
                # print("mean_value:",mean_value)

                # max_index = v.index(max_value)
                # print("max_index:", max_index)
                temp_auc_mean[temp_auc_mean_key].append(mean_value)
                # print("model_value["+classif_str+"_Accuracy][k]:", model_value[classif_str+"_Accuracy"][k])
                temp_auc_mean_corres_acc_index[temp_auc_mean_key].append(statistics.mean(model_value[classif_str + corres_eval_metric][k]))
                temp_auc_mean_feat_list[temp_auc_mean_key].append(k)
                # print("value:", model_value[classif_str+corres_eval_metric][k])
                # print("value:", k)

            prev_auc = 0.0
            # print("key:", key)
            # print("temp_auc_mean:", temp_auc_mean)
            for k,v in temp_auc_mean.items():

                mean_values_list = zip(v, temp_auc_mean_feat_list[k], temp_auc_mean_corres_acc_index[k])
                # print(mean_values_list)
                mean_values_list.sort(key=lambda x:x[0], reverse=True)
                # print()
                # print(mean_values_list[:5])
                all_classif_auc[classif_str] = all_classif_auc.get(classif_str, []) + mean_values_list[:5]
            try:
                final_features = ""
                final_auc_50 = []
                if classifiers_str.index(classif_str) == 0:
                    final_features, final_auc_50 = run_50_fold_return_low_var_feature_mlp(all_classif_auc[classif_str], is_accuracy)
                if classifiers_str.index(classif_str) == 1:
                    final_features, final_auc_50 = run_50_fold_return_low_var_feature_svm(all_classif_auc[classif_str], is_accuracy)
                if classifiers_str.index(classif_str) == 2:
                    final_features, final_auc_50 = run_50_fold_return_low_var_feature_rndf(all_classif_auc[classif_str], is_accuracy)

                top_5_mean_50_fold_classif_result[key][classif_str + eval_metric][final_features] = final_auc_50
            except Exception as e:
                pickle_out = open("temp_top_5_mean_50_fold_classif_result.pickle", "wb")
                pickle.dump(top_5_mean_50_fold_classif_result, pickle_out)
                pickle_out.close()
                print("---error--")
                print(e)

    pickle_out = open("top_5_mean_50_fold_classif_result.pickle", "wb")
    pickle.dump(top_5_mean_50_fold_classif_result, pickle_out)
    pickle_out.close()
    print("perform_repeated_k_fold. time taken: ", time.strftime("%H:%M:%S", time.gmtime(time.time() - start)))




def run_50_fold_return_low_var_feature_mlp(comb_list, accurracy):
    X, y, test_dataset = readDataFromFile()
    prev_std = 999999999999
    final_feat_comb = ""
    final_auc_result = []
    for comb in comb_list:
        X_slice = X[comb[1].split(", ")]
        n_splits = 5
        n_repeats = 10
        kf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=None)

        mlp_auc_result = []
        # svm_auc_result = []
        # rndf_auc_result = []

        for train_index, test_index in kf.split(X_slice):
            # print("Train:", len(train_index), "Validation:", len(test_index))
            X_train, X_test = X_slice.iloc[train_index], X_slice.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            scaler = StandardScaler()

            mlp_X_train = scaler.fit_transform(X_train)
            mlp_X_test = scaler.transform(X_test)

            # svm_X_train = scaler.fit_transform(X_train[svm_feat_list])
            # svm_X_test = scaler.transform(X_test[svm_feat_list])

            # rndf_X_train = scaler.fit_transform(X_train[rndf_feat_list])
            # rndf_X_test = scaler.transform(X_test[rndf_feat_list])

            mlp_data_dict = multi_layer_perceptron_classifier(mlp_X_train, y_train, mlp_X_test, test_dataset, y_test,
                                                              cross_validation=True)

            # svm_data_dict = run_svm(svm_X_train, y_train, svm_X_test, y_test)

            # rf_data_dict = random_forest_classifier(rndf_X_train, y_train, rndf_X_test, y_test, test_dataset, True)

            if not accurracy:
                mlp_auc_result.append(round(float(mlp_data_dict["AUC"]), 4))
                # svm_auc_result.append(round(float(svm_data_dict["AUC"]), 4))
                # rndf_auc_result.append(round(float(rf_data_dict["AUC"]), 4))

            if accurracy:
                mlp_auc_result.append(round(float(mlp_data_dict["Accuracy"]), 4))
                # svm_auc_result.append(round(float(svm_data_dict["Accuracy"]), 4))
                # rndf_auc_result.append(round(float(rf_data_dict["Accuracy"]), 4))

        new_std = np.std(mlp_auc_result)
        if new_std < prev_std:
            prev_std = new_std
            final_feat_comb = comb[1]
            final_auc_result = mlp_auc_result

    return final_feat_comb, final_auc_result


def run_50_fold_return_low_var_feature_svm(comb_list, accurracy):
    X, y, test_dataset = readDataFromFile()
    prev_std = 999999999999
    final_feat_comb = ""
    final_auc_result = []
    for comb in comb_list:
        X_slice = X[comb[1].split(", ")]
        n_splits = 5
        n_repeats = 10
        kf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=None)

        svm_auc_result = []

        for train_index, test_index in kf.split(X_slice):
            # print("Train:", len(train_index), "Validation:", len(test_index))
            X_train, X_test = X_slice.iloc[train_index], X_slice.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            scaler = StandardScaler()

            svm_X_train = scaler.fit_transform(X_train)
            svm_X_test = scaler.transform(X_test)


            svm_data_dict = run_svm(svm_X_train, y_train, svm_X_test, y_test)

            if not accurracy:
                svm_auc_result.append(round(float(svm_data_dict["AUC"]), 4))

            if accurracy:
                svm_auc_result.append(round(float(svm_data_dict["Accuracy"]), 4))

        new_std = np.std(svm_auc_result)
        if new_std < prev_std:
            prev_std = new_std
            final_feat_comb = comb[1]
            final_auc_result = svm_auc_result

    return final_feat_comb, final_auc_result


def run_50_fold_return_low_var_feature_rndf(comb_list, accurracy):
    X, y, test_dataset = readDataFromFile()
    prev_std = 999999999999
    final_feat_comb = ""
    final_auc_result = []
    for comb in comb_list:
        X_slice = X[comb[1].split(", ")]
        n_splits = 5
        n_repeats = 10
        kf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=None)

        rndf_auc_result = []

        for train_index, test_index in kf.split(X_slice):
            # print("Train:", len(train_index), "Validation:", len(test_index))
            X_train, X_test = X_slice.iloc[train_index], X_slice.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            scaler = StandardScaler()

            rndf_X_train = scaler.fit_transform(X_train)
            rndf_X_test = scaler.transform(X_test)

            rf_data_dict = random_forest_classifier(rndf_X_train, y_train, rndf_X_test, y_test, test_dataset, True)

            if not accurracy:
                rndf_auc_result.append(round(float(rf_data_dict["AUC"]), 4))

            if accurracy:
                rndf_auc_result.append(round(float(rf_data_dict["Accuracy"]), 4))

        new_std = np.std(rndf_auc_result)
        if new_std < prev_std:
            prev_std = new_std
            final_feat_comb = comb[1]
            final_auc_result = rndf_auc_result

    return final_feat_comb, final_auc_result






def fold_50_t_test(mlp_feat_list, svm_feat_list, rndf_feat_list, accurracy=False):
    X, y, test_dataset = readDataFromFile()
    n_splits = 5
    n_repeats = 10
    kf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=None)

    mlp_auc_result = []
    svm_auc_result = []
    rndf_auc_result = []

    mlp_accuracy_result = []
    svm_accuracy_result = []
    rndf_accuracy_result = []

    mlp_sensitivity_result = []
    svm_sensitivity_result = []
    rndf_sensitivity_result = []

    mlp_specificity_result = []
    svm_specificity_result = []
    rndf_specificity_result = []

    for train_index, test_index in kf.split(X):
        print()
        # print("Train:", len(train_index), "Validation:", len(test_index))
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        scaler = StandardScaler()

        mlp_X_train = scaler.fit_transform(X_train[mlp_feat_list])
        mlp_X_test = scaler.transform(X_test[mlp_feat_list])

        svm_X_train = scaler.fit_transform(X_train[svm_feat_list])
        svm_X_test = scaler.transform(X_test[svm_feat_list])

        rndf_X_train = scaler.fit_transform(X_train[rndf_feat_list])
        rndf_X_test = scaler.transform(X_test[rndf_feat_list])


        mlp_data_dict = multi_layer_perceptron_classifier(mlp_X_train, y_train, mlp_X_test, test_dataset, y_test,
                                                          cross_validation=True)

        svm_data_dict = run_svm(svm_X_train, y_train, svm_X_test, y_test)

        rf_data_dict = random_forest_classifier(rndf_X_train, y_train, rndf_X_test, y_test, test_dataset, True)

        mlp_auc_result.append(round(float(mlp_data_dict["AUC"]), 3))
        svm_auc_result.append(round(float(svm_data_dict["AUC"]), 3))
        rndf_auc_result.append(round(float(rf_data_dict["AUC"]), 3))

        mlp_accuracy_result.append(round(float(mlp_data_dict["Accuracy"]), 3))
        svm_accuracy_result.append(round(float(svm_data_dict["Accuracy"]), 3))
        rndf_accuracy_result.append(round(float(rf_data_dict["Accuracy"]), 3))

        mlp_sensitivity_result.append(round(float(mlp_data_dict["Sensitivity"]), 3))
        svm_sensitivity_result.append(round(float(svm_data_dict["Sensitivity"]), 3))
        rndf_sensitivity_result.append(round(float(rf_data_dict["Sensitivity"]), 3))

        mlp_specificity_result.append(round(float(mlp_data_dict["Specificity"]), 3))
        svm_specificity_result.append(round(float(svm_data_dict["Specificity"]), 3))
        rndf_specificity_result.append(round(float(rf_data_dict["Specificity"]), 3))

    mlp_auc_result.sort()
    svm_auc_result.sort()
    rndf_auc_result.sort()

    mlp_accuracy_result.sort()
    svm_accuracy_result.sort()
    rndf_accuracy_result.sort()

    print()
    print_mean_min_max(mlp_auc_result, svm_auc_result, rndf_auc_result, "auc")

    print_result_of_t_test_pairwise_comparison(mlp_auc_result, svm_auc_result, rndf_auc_result, "auc")

    mlp_txt = "Neural Net: " + ", ".join(mlp_feat_list)
    svm_txt = "SVM: " + ", ".join(svm_feat_list)
    rndf_txt = "RndF: " + ", ".join(rndf_feat_list)


    plot_mlp_auc_result = get_sorted_count_orddict_from_list(mlp_auc_result)
    plot_svm_auc_result = get_sorted_count_orddict_from_list(svm_auc_result)
    plot_rndf_auc_result = get_sorted_count_orddict_from_list(rndf_auc_result)

    # plt.suptitle("Distribution")

    mlp_pdf = get_pdf_for_points(mlp_auc_result)
    print("plot_mlp_auc_result:", list(plot_mlp_auc_result.keys()), " - ", list(plot_mlp_auc_result.values()))

    svm_pdf = get_pdf_for_points(svm_auc_result)
    print("plot_svm_auc_result:", list(plot_svm_auc_result.keys()), " - ", list(plot_svm_auc_result.values()))

    rndf_pdf = get_pdf_for_points(rndf_auc_result)
    print("plot_rndf_auc_result:", list(plot_rndf_auc_result.keys()), " - ", list(plot_rndf_auc_result.values()))


    xlabel = "AUC"
    ylabel = "PDF(Probability Density Function)"
    plot_graph([mlp_auc_result, mlp_pdf], [svm_auc_result, svm_pdf], [rndf_auc_result, rndf_pdf], mlp_txt, svm_txt, rndf_txt, xlabel, ylabel)

    print()
    print_mean_min_max(mlp_accuracy_result, svm_accuracy_result, rndf_accuracy_result, "accuracy")
    print_result_of_t_test_pairwise_comparison(mlp_accuracy_result, svm_accuracy_result, rndf_accuracy_result, "accuracy")

    plot_mlp_accuracy_result = get_sorted_count_orddict_from_list(mlp_accuracy_result)
    plot_svm_accuracy_result = get_sorted_count_orddict_from_list(svm_accuracy_result)
    plot_rndf_accuracy_result = get_sorted_count_orddict_from_list(rndf_accuracy_result)

    # plt.suptitle("Distribution")

    mlp_pdf = get_pdf_for_points(mlp_accuracy_result)
    print("plot_mlp_accuracy_result:", list(plot_mlp_accuracy_result.keys()), " - ", list(plot_mlp_accuracy_result.values()))

    svm_pdf = get_pdf_for_points(svm_accuracy_result)
    print("plot_svm_accuracy_result:", list(plot_svm_accuracy_result.keys()), " - ", list(plot_svm_accuracy_result.values()))

    rndf_pdf = get_pdf_for_points(rndf_accuracy_result)
    print("plot_rndf_accuracy_result:", list(plot_rndf_accuracy_result.keys()), " - ", list(plot_rndf_accuracy_result.values()))

    xlabel = "ACCURACY"
    plot_graph([mlp_accuracy_result, mlp_pdf], [svm_accuracy_result, svm_pdf], [rndf_accuracy_result, rndf_pdf], mlp_txt, svm_txt, rndf_txt, xlabel, ylabel)

    print()
    print_mean_min_max(mlp_sensitivity_result, svm_sensitivity_result, rndf_sensitivity_result, "sensitivity")
    print()
    print_mean_min_max(mlp_specificity_result, svm_specificity_result, rndf_specificity_result, "specificity")


    # ylabel = "COUNT"
    # plot_graph([list(plot_mlp_auc_result.keys()), list(plot_mlp_auc_result.values())],
    #            [list(plot_svm_auc_result.keys()), list(plot_svm_auc_result.values())],
    #            [list(plot_rndf_auc_result.keys()), list(plot_rndf_auc_result.values())], mlp_txt, svm_txt, rndf_txt, xlabel, ylabel)


def print_mean_min_max(mlp_result, svm_result, rndf_result, str_type):
    print("\n----MLP----")
    print("mlp_"+str_type+"_result:", mlp_result)
    print("mean:", statistics.mean(mlp_result))
    print("min:", min(mlp_result))
    print("max:", max(mlp_result))

    print("\n----SVM----")
    print("svm_"+str_type+"_result:", svm_result)
    print("mean:", statistics.mean(svm_result))
    print("min:", min(svm_result))
    print("max:", max(svm_result))

    print("\n----RNDF----")
    print("rndf_"+str_type+"_result:", rndf_result)
    print("mean:", statistics.mean(rndf_result))
    print("min:", min(rndf_result))
    print("max:", max(rndf_result))

def print_result_of_t_test_pairwise_comparison(mlp_result, svm_result, rndf_result, str_type):
    print("Classifiers pairwise t-test result of "+str_type)
    mlp_svm_results = scipy.stats.ttest_ind(mlp_result, svm_result)
    print("mlp_svm_results:", mlp_svm_results," - statistic=", round(mlp_svm_results[0], 9), ", pvalue=", round(mlp_svm_results[1], 9))
    mlp_rndf_results = scipy.stats.ttest_ind(mlp_result, rndf_result)
    print("mlp_rndf_results:", mlp_rndf_results,"  - statistic=", round(mlp_rndf_results[0], 9), ", pvalue=", round(mlp_rndf_results[1], 9))
    rndf_svm_results = scipy.stats.ttest_ind(rndf_result, svm_result)
    print("rndf_svm_results:", rndf_svm_results,"  - statistic=", round(rndf_svm_results[0], 9), ", pvalue=", round(rndf_svm_results[1], 9))

def get_pdf_for_points(points):
    pdf = scipy.stats.norm.pdf(points, np.mean(points), np.std(points))
    return pdf

def get_sorted_count_orddict_from_list(points):
    result = OrderedDict(sorted(Counter(points).items(), key=lambda t: t[0]))
    return result



def plot_graph(mlp_coord, svm_coord, rndf_coord, mlp_txt, svm_txt, rndf_txt, xlabel, ylabel):
    plt.plot(mlp_coord[0], mlp_coord[1], color='xkcd:cherry', marker='o', label='Neural Net')
    # plt.plot(mlp_auc_result, pdf, color='xkcd:cherry', marker='o', label='Neural Net')

    plt.plot(svm_coord[0], svm_coord[1], color='xkcd:royal blue', marker='o',
             label='SVM')
    # plt.plot(svm_auc_result, pdf, color='xkcd:royal blue', marker='o', label='SVM')

    plt.plot(rndf_coord[0], rndf_coord[1], color='xkcd:green', marker='o',
             label='RndF')
    # plt.plot(rndf_auc_result, pdf, color='xkcd:green', marker='o', label='RndF')

    plt.text(0.12, 0.13, mlp_txt, fontsize=10, transform=plt.gcf().transFigure)
    plt.text(0.12, 0.09, svm_txt, fontsize=10, transform=plt.gcf().transFigure)
    plt.text(0.12, 0.05, rndf_txt, fontsize=10, transform=plt.gcf().transFigure)
    plt.subplots_adjust(bottom=0.25)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

def actual_test_set_result(mlp_feat_list, svm_feat_list, rndf_feat_list):
    X, y, test_dataset = readDataFromFile()

    data = pd.read_csv("child_growth_dataset/weighted_knn_imputed_test_data.csv")
    test_dataset = []
    y_test = data.iloc[:, data.columns.get_loc('stunting36')]
    data.drop(['stunting36'], axis=1, inplace=True)
    X_test = data.iloc[:, :]  # independent columns
    print(X_test.head())
    results = get_model_results(X, y, X_test, y_test, None, "")
    print(results)






def merge_feat_imp_list(old_feat, new_feat, iter_val, num_of_fold):
    iter_val += 1
    updated_feat_list = dict(old_feat)
    for feat, value in new_feat:
        updated_feat_list[feat] = updated_feat_list.get(feat, 0) + value

    if iter_val == num_of_fold:
        updated_feat_list = [(k, v/num_of_fold) for k, v in updated_feat_list.items()]
    else:
        updated_feat_list = list(updated_feat_list.items())

    return updated_feat_list


def merge_feat_count_list(old_feat, new_feat):
    for feat in new_feat:
        old_data = list(filter(lambda item:item[0] == feat, old_feat))
        if old_data:
            old_feat_index = old_feat.index(old_data[0])
            old_feat[old_feat_index] = (feat, old_feat[old_feat_index][1] + 1)
        else:
            old_feat.append((feat, 1))
    return old_feat


def get_model_results(X_train, y_train, X_test, y_test, feat_list, method_type):
    # Feature Scaling
    if feat_list == [] or feat_list == "":
        return []
    test_dataset = []

    # print(feat_list)
    # print(X_train.head())
    # print(X_test['inflenght_36month'])
    if not isinstance(feat_list, list) and feat_list is not None:
        feat_list = feat_list.split(", ")
        X_test = X_test[feat_list]


    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    temp_dict = {"method": method_type, "Features": feat_list}
    mlp_data_dict = multi_layer_perceptron_classifier(X_train, y_train, X_test, test_dataset, y_test,
                                                      cross_validation=True)
    mlp_data_dict.update(temp_dict)
    svm_data_dict = run_svm(X_train, y_train, X_test, y_test)
    svm_data_dict.update(temp_dict)
    rf_data_dict = random_forest_classifier(X_train, y_train, X_test, y_test, test_dataset, True)
    rf_data_dict.update(temp_dict)

    return [mlp_data_dict, svm_data_dict, rf_data_dict]


def get_all_classifier_result():
    X, y, test_dataset = readDataFromFile()
    print("------------------all feature result----------------")
    all_features_result = crossValidation(X, y, "all", "none")

    print("------------------feature_mutual_info result----------------")
    X, y, X_test, feat_list, _ = feature_mutual_info(X, y)
    mutual_info_result = crossValidation(X, y, feat_list, "mutual information")

    field_names = list(all_features_result[0].keys())
    print("Generating result CSV")
    generate_csv_from_dict(dataset=mutual_info_result, fieldnames=field_names, file_name="mutual_info_result")
    print("Done.")

    # print("------------------feature_selectkbest chi2 result----------------")
    # X, y, X_test, feat_list, _ = feature_selectkbest_chi2(X, y)
    # selectkbest_chi2_result = crossValidation(X, y, feat_list, "selectkbest chi2")

    return ""

    print("------------------feature importance result----------------")
    X, y, X_test, feat_list, _ = FeatureImportance(X, y)
    features_imp_result = crossValidation(X, y, feat_list, "feature importance")

    print("------------------L1_based_feature_selection result----------------")
    X, y, X_test, feat_list = L1_based_feature_selection(X, y)
    L1_based_feature_selection_result = crossValidation(X, y, feat_list, "L1 based feature selectione")


    print("------------------feature_selectkbest result----------------")
    X, y, X_test, feat_list, _ = feature_selectkbest_f_classif(X, y)
    selectkbest_f_classif_result = crossValidation(X, y, feat_list, "selectkbest f_classif")

    print("------------------feature_GenericUnivariateSelect result----------------")
    X, y, X_test, feat_list, _ = feature_GenericUnivariateSelect(X, y)
    GenericUnivariateSelect_result = crossValidation(X, y, feat_list, "GenericUnivariateSelect")

    print("------------------feature_SelectPercentile result----------------")
    X, y, X_test, feat_list, _ = feature_SelectPercentile(X, y)
    SelectPercentile_result = crossValidation(X, y, feat_list, "SelectPercentile")

    print("------------------feature_SelectFpr result----------------")
    X, y, X_test, feat_list, _ = feature_SelectFpr(X, y)
    selectFpr_result = crossValidation(X, y, feat_list, "selectFpr")

    print("------------------feature_SelectFdr result----------------")
    X, y, X_test, feat_list, _ = feature_SelectFdr(X, y)
    selectFdr_result = crossValidation(X, y, feat_list, "selectFdr")

    print("------------------feature_RFE result----------------")
    X, y, X_test, feat_list = feature_RFE(X, y)
    RFE_result = crossValidation(X, y, feat_list, "Recursive feature elimination")

    print("------------------feature_RFECV result----------------")
    X, y, X_test, feat_list = feature_RFECV(X, y)
    RFECV_result = crossValidation(X, y, feat_list, "Recursive feature elimination and cross-validated selection")


    all_data = all_features_result + features_imp_result + L1_based_feature_selection_result + selectFpr_result + selectFdr_result + selectkbest_f_classif_result +\
               GenericUnivariateSelect_result + SelectPercentile_result + mutual_info_result + RFE_result + RFECV_result
    field_names = list(all_features_result[0].keys())
    print("Generating result CSV")
    generate_csv_from_dict(dataset=all_data, fieldnames=field_names, file_name="all_results_knn")
    print("Done.")


def plot_top_5_mean_50_fold_classif_result():
    pickle_in = open("top_5_mean_50_fold_classif_result_Accuracy.pickle", "rb")
    results_dict = pickle.load(pickle_in)
    # mlp_str = "mlp_AUC"
    # svm_str = "svm_AUC"
    # rndf_str = "rndf_AUC"

    mlp_str = "mlp_Accuracy"
    svm_str = "svm_Accuracy"
    rndf_str = "rndf_Accuracy"

    mlp_prev_mean = 0
    mlp_feat_list = ""
    mlp_result_list = []

    svm_prev_mean = 0
    svm_feat_list = ""
    svm_result_list = []

    rndf_prev_mean = 0
    rndf_feat_list = ""
    rndf_result_list = []
    for key, value in results_dict.items():

        # print(results_dict[key]["mlp_AUC"].items()[0][1])

        mlp_new_mean = statistics.mean(results_dict[key][mlp_str].items()[0][1])
        if mlp_new_mean > mlp_prev_mean:
            print("MLP - ", key, " - ", mlp_new_mean)
            mlp_prev_mean = mlp_new_mean
            mlp_feat_list = results_dict[key][mlp_str].items()[0][0]
            mlp_result_list = [round(val, 2) for val in results_dict[key][mlp_str].items()[0][1]]

        svm_new_mean = statistics.mean(results_dict[key][svm_str].items()[0][1])
        if svm_new_mean > svm_prev_mean:
            print("SVM - ", key, " - ", svm_new_mean)
            svm_prev_mean = svm_new_mean
            svm_feat_list = results_dict[key][svm_str].items()[0][0]
            svm_result_list = [round(val, 2) for val in results_dict[key][svm_str].items()[0][1]]
            # svm_result_list = results_dict[key]["svm_AUC"].items()[0][1]

        rndf_new_mean = statistics.mean(results_dict[key][rndf_str].items()[0][1])
        if rndf_new_mean > rndf_prev_mean:
            print("RNDF - ", key, " - ", rndf_new_mean)
            rndf_prev_mean = rndf_new_mean
            rndf_feat_list = results_dict[key][rndf_str].items()[0][0]
            rndf_result_list = [round(val, 2) for val in results_dict[key][rndf_str].items()[0][1]]
            # rndf_result_list = results_dict[key]["rndf_AUC"].items()[0][1]

    # print("feat_list:", feat_list)


    mlp_result_list.sort()
    svm_result_list.sort()
    rndf_result_list.sort()

    print("\nmlp_result_list:", mlp_feat_list.split(", "))
    print("\nsvm_result_list:", svm_feat_list.split(", "))
    print("\nrndf_result_list:", rndf_feat_list.split(", "))
    print("Union:", list(set(mlp_feat_list.split(", ") + svm_feat_list.split(", ") + rndf_feat_list.split(", "))))

    plot_mlp_auc_result = OrderedDict(sorted(Counter(mlp_result_list).items(), key=lambda t: t[0]))
    # sorted(plot_mlp_auc_result.items(), key=itemgetter(0))
    plot_svm_auc_result = OrderedDict(sorted(Counter(svm_result_list).items(), key=lambda t: t[0]))
    # sorted(plot_svm_auc_result.items(), key=itemgetter(0))
    plot_rndf_auc_result = OrderedDict(sorted(Counter(rndf_result_list).items(), key=lambda t: t[0]))

    mlp_txt = "Neural Net: " + mlp_feat_list
    svm_txt = "SVM: " + svm_feat_list
    rndf_txt = "RndF: " + rndf_feat_list

    # plt.suptitle("Distribution")

    pdf = scipy.stats.norm.pdf(mlp_result_list, np.mean(mlp_result_list), np.std(mlp_result_list))
    # print("plot_mlp_auc_result:", list(plot_mlp_auc_result.keys()), " - ", list(plot_mlp_auc_result.values()))
    # plt.plot(list(plot_mlp_auc_result.keys()), list(plot_mlp_auc_result.values()), color='xkcd:cherry', marker='o', label='Neural Net')
    plt.plot(mlp_result_list, pdf, color='xkcd:cherry', marker='o', label='Neural Net')

    pdf = scipy.stats.norm.pdf(svm_result_list, np.mean(svm_result_list), np.std(svm_result_list))
    # print("plot_svm_auc_result:", list(plot_svm_auc_result.keys()), " - ", list(plot_svm_auc_result.values()))
    # plt.plot(list(plot_svm_auc_result.keys()), list(plot_svm_auc_result.values()), color='xkcd:royal blue', marker='o', label='SVM')
    plt.plot(svm_result_list, pdf, color='xkcd:royal blue', marker='o', label='SVM')

    pdf = scipy.stats.norm.pdf(rndf_result_list, np.mean(rndf_result_list), np.std(rndf_result_list))
    # print("plot_rndf_auc_result:", list(plot_rndf_auc_result.keys()), " - ", list(plot_rndf_auc_result.values()))
    # plt.plot(list(plot_rndf_auc_result.keys()), list(plot_rndf_auc_result.values()), color='xkcd:green', marker='o', label='RndF')
    plt.plot(rndf_result_list, pdf, color='xkcd:green', marker='o', label='RndF')

    plt.text(0.12, 0.13, mlp_txt, fontsize=10, transform=plt.gcf().transFigure)
    plt.text(0.12, 0.09, svm_txt, fontsize=10, transform=plt.gcf().transFigure)
    plt.text(0.12, 0.05, rndf_txt, fontsize=10, transform=plt.gcf().transFigure)
    # plt.xlabel("AUC")
    plt.xlabel("ACCURACY")
    plt.ylabel("PDF(Probability Density Function)")
    # plt.ylabel("COUNT")
    plt.subplots_adjust(bottom=0.25)
    plt.legend()
    plt.show()


def violin_swarm_plot():

    X, y, test_dataset = readDataFromFile()
    data_dia = y
    data = X
    data_n_2 = (data - data.mean()) / (data.std())  # standardization
    print(X.shape[1])

    next_idx = 0
    # feature_len = 15

    while next_idx <= X.shape[1]:
        prev_idx = next_idx
        next_idx += 15
        next_idx = next_idx if next_idx < X.shape[1] else X.shape[1]
        print("range: ",prev_idx, " to ", next_idx)


        data = pd.concat([y, data_n_2.iloc[:, prev_idx:next_idx]], axis=1)
        data = pd.melt(data, id_vars="stunting36",
                       var_name="features",
                       value_name='value')
        plt.figure(figsize=(12, 8))
        sns.violinplot(x="features", y="value", hue="stunting36", data=data, split=True, inner="quart")
        plt.xticks(rotation=90)
        # plt.legend()
        plt.show()



        # sns.set(style="whitegrid", palette="muted")
        # data = pd.concat([y, data_n_2.iloc[:, prev_idx:next_idx]], axis=1)
        # data = pd.melt(data, id_vars="stunting36",
        #                var_name="features",
        #                value_name='value')
        # plt.figure(figsize=(12, 8))
        # # tic = time.time()
        # sns.swarmplot(x="features", y="value", hue="stunting36", data=data)
        # print("show")
        #
        # plt.xticks(rotation=90)
        # plt.show()


def heat_map(feat_list):
    X, y, test_dataset = readDataFromFile()
    # correlation map
    X = X[feat_list]
    f, ax = plt.subplots(figsize=(10, 10))
    plt.rcParams['figure.figsize'] = (10, 10)
    sns.heatmap(X.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)
    plt.show()



def multi_layer_perceptron_classifier(X_train, y_train, X_test, test_dataset, y_test=None, cross_validation=False):
    print("---------------------multi_layer_perceptron_classifier--------------------------")
    # classifier = MLPClassifier(activation="tanh", solver='sgd', alpha=1e-5, hidden_layer_sizes=(14, 14, 14), random_state=1, max_iter=500)
    # print("Building model")
    classifier = MLPClassifier(hidden_layer_sizes=(50, 50, 50), max_iter=500)
    # print(classifier)
    classifier.fit(X_train, y_train)
    # print("Building model done")
    # print("Feature Importance values:", classifier.feature_importances_)
    # feature_importances = pd.DataFrame(classifier.feature_importances_,
    #                                    index=test_dataset.columns[2:],
    #                                    columns=['importance']).sort_values('importance', ascending=False)

    # print("Feature Importance:", feature_importances)
    # Predicting the Test set results
    # print("Predicting")
    # y_pred_proba = classifier.predict_proba(X_test)[:, 1]
    y_pred = classifier.predict(X_test)
    # print("Predicting done")
    # print(classification_report(y_test, y_pred))
    # print(y_pred.shape)
    # print(y_pred)
    # print(y_pred_proba.shape)
    # print(y_pred_proba)
    # print(type(y_pred_proba))

    if not cross_validation:
        print("Generating CSV")
        # generate_result_csv_from_prediction(y_pred_proba, file_name="test_result_multi_layer_perceptron")
        generate_result_csv_from_prediction(y_pred, file_name="test_result_multi_layer_perceptron")

    if cross_validation:
        result = evaluation_metrics(y_test, y_pred)
        data_dict = {"Classifier": "Multi Layer Perceptron"}
        data_dict.update(result)
        return data_dict
        # print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
        # print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
        # print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
        #
        # print("accuracy_score:", metrics.accuracy_score(y_test, y_pred))

    print("---------------------multi_layer_perceptron_classifier Done--------------------------")


def run_svm(x_train,y_train,x_test,y_test):
    print("---------------------SVM_classifier--------------------------")
    # kf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=None)
    # for train_index, test_index in kf.split(x_data):
    #     # print("Train:", train_index, "Validation:", test_index)
    #     x_train, x_test = x_data[train_index], x_data[test_index]
    #     y_train, y_test = y_data[train_index], y_data[test_index]

    clf = SVC(gamma='auto', probability=True)
    # print("Model config: ")
    # print(clf)
    clf.fit(x_train, y_train)
    rfr_predictions = clf.predict(x_test)
    rfr_prediction_proba = clf.predict_proba(x_test)[:, 0]
    # print("prediction complete")

    # Calculating Evaluating Metrics
    result = evaluation_metrics(y_test, rfr_predictions)
    data_dict = {"Classifier": "Support Vector Machine"}
    data_dict.update(result)
    return data_dict
    # print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, rfr_predictions))
    # print('Mean Squared Error:', metrics.mean_squared_error(y_test, rfr_predictions))
    # print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, rfr_predictions)))
    #
    # print("accuracy_score:", metrics.accuracy_score(y_test, rfr_predictions))
    # print("roc_auc_score:", roc_auc_score(y_test, rfr_prediction_proba))

    # field_names = ["Id", "Prediction", "Proba"]
    # rfr_test_result = []
    #
    # i = 0
    # for x, y in zip(np.nditer(rfr_predictions), np.nditer(rfr_prediction_proba)):
    #     rfr_test_result.append({"Id": i, "Prediction": x, "Proba": y})
    #     i += 1
    #
    # print('creating svm csv file')
    # with open('predict_svm.csv', 'w', newline='') as csvfile:
    #     writer = csv.DictWriter(csvfile, fieldnames=field_names)
    #     writer.writeheader()
    #     writer.writerows(rfr_test_result)
    print("---------------------SVM_classifier done--------------------------")


def random_forest_classifier(X_train, y_train, X_test, y_test, test_dataset, validation):
    print("---------------------RandomForestClassifier--------------------------")
    # Fitting Random Forest Classification to the Training set
    classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=42)
    classifier.fit(X_train, y_train)

    # feature_importances = pd.DataFrame(classifier.feature_importances_,
    #                                    index=test_dataset.columns[2:],
    #                                    columns=['importance']).sort_values('importance', ascending=False)
    #
    # print("Feature Importance:", feature_importances)
    # Predicting the Test set results
    y_pred_proba = classifier.predict_proba(X_test)[:, 1]
    y_pred = classifier.predict(X_test)

    if not validation:
        generate_result_csv_from_prediction(y_pred_proba, file_name="test_result_random_forest")

    if validation:
        result = evaluation_metrics(y_test, y_pred)
        data_dict = {"Classifier": "Random Forest(entropy)"}
        data_dict.update(result)
        return data_dict

    print("---------------------RandomForestClassifier Done--------------------------")


def crossValidation(X, y, feat_list, method_type):
    # Creating the Training and Test set from data
    # X, y, temp = readDataFromFile()
    # X, y, temp = FeatureImportance()
    test_dataset = []
    # print(X.head())
    # print(y.head())

    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.25, random_state=21)
    # Feature Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    temp_dict = {"method": method_type, "Features": feat_list}
    mlp_data_dict = multi_layer_perceptron_classifier(X_train, y_train, X_test, test_dataset, y_test, cross_validation=True)
    mlp_data_dict.update(temp_dict)
    svm_data_dict = run_svm(X_train, y_train, X_test, y_test)
    svm_data_dict.update(temp_dict)
    rf_data_dict = random_forest_classifier(X_train, y_train, X_test, y_test, test_dataset, True)
    rf_data_dict.update(temp_dict)

    return [mlp_data_dict, svm_data_dict, rf_data_dict]




def FeatureImportance(X_train, X_raw, y):
    # data = pd.read_csv("santander-customer-transaction-prediction/train.csv")
    # X = data.iloc[:, 2:]  # independent columns
    # y = data.iloc[:, 1]  # target column
    # X, y, test_dataset = readDataFromFile()
    # X_test = test_dataset.iloc[:, 1:]
    # print(type(X))
    # print(X.head())
    # print(y.head())
    #
    model = ExtraTreesClassifier()
    model.fit(X_train, y)
    # print(model.feature_importances_)  # use inbuilt class feature_importances of tree based classifiers
    # plot graph of feature importances for better visualization
    feat_importances = pd.Series(model.feature_importances_, index=X_raw.columns).nlargest(20)
    feat_list = [str(feat_importances.index[i]) for i in range(0, len(feat_importances))]
    feat_list_with_imp = [(str(feat_importances.index[i]), round(feat_importances[i], 7)) for i in range(0, len(feat_importances))]
    X = X_raw[feat_list]
    y = y
    # X_test = X_test[feat_list]
    X_test = []

    # feat_importances.plot(kind='barh')
    # plt.show()
    return X, y, X_test, ', '.join(feat_list), feat_list_with_imp



def L1_based_feature_selection(X_train, X_raw, y):
    print("Running L1_based_feature_selection")
    # X, y, temp = readDataFromFile()
    # print(X.shape)
    # print(X.columns.values.tolist())

    alpha_lasso = [1e-15, 1e-10, 1e-8, 1e-5, 1e-4, 1e-3, 1e-2, 1, 5, 10]
    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X_train, y)
    model = SelectFromModel(lsvc, prefit=True)
    X_new = model.transform(X_train)
    # print(model.ranking_)
    # print(model.get_support())
    # print(model.get_support(indices=True))
    feat_importances = pd.Series(model.get_support(), index=X_raw.columns)
    feat_list = []
    for col in X_raw.columns:
        if feat_importances[col]:
            feat_list.append(col)
    # print(feat_list)

    # feat_importances = pd.Series(model.ranking_, index=X.columns).nlargest(20)
    # feat_list = [feat_importances.index[i] for i in range(0, len(feat_importances))]
    # print(feat_list)

    # # print(X.columns.values.tolist())
    # X_train, X_test, y_train, y_test = train_test_split(X_new.values, y.values, test_size=0.25, random_state=21)
    # # Feature Scaling
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)
    #
    # multi_layer_perceptron_classifier(X_train, y_train, X_test, test_dataset, y_test, cross_validation=True)
    feat_list_with_imp = [(str(feat_importances.index[i]), 1) for i in
                          range(0, len(feat_importances))]
    X = X_raw[feat_list]
    # X_test = X_test[feat_list]
    X_test = []

    print("Completed L1_based_feature_selection")
    return X, y, X_test, ', '.join(feat_list), feat_list_with_imp



def feature_SelectFpr(X_train, X_raw, y_data):
    print("--------------------------------Running SelectFpr")
    bestfeatures = SelectFpr(f_classif, alpha=0.01)
    fit = bestfeatures.fit(X_train, y_data)
    x_new = bestfeatures.transform(X_train)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X_raw.columns)
    # concat two dataframes for better visualization

    feat_importances = pd.Series(fit.scores_, index=X_raw.columns).nlargest(20)
    # print(feat_importances)
    feat_list = [feat_importances.index[i] for i in range(0, len(feat_importances))]
    feat_list_with_imp = [(str(feat_importances.index[i]), round(feat_importances[i], 7)) for i in
                          range(0, len(feat_importances))]
    # print(feat_list_with_imp)

    # featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    # featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    # print("SelectFpr features")
    # print(featureScores.nlargest(10, 'Score'))  # print 10 best features

    X = X_raw[feat_list]
    y = y_data
    # X_test = X_test[feat_list]
    X_test = []

    # feat_importances.plot(kind='barh')
    # plt.show()
    print("Completed SelectFpr")
    return X, y, X_test, ', '.join(feat_list), feat_list_with_imp


def feature_SelectFdr(X_train, X_raw, y_data):
    print("Running SelectFdr")
    bestfeatures = SelectFdr(f_classif, alpha=0.01)
    fit = bestfeatures.fit(X_train, y_data)
    x_new = bestfeatures.transform(X_train)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X_raw.columns)
    # concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    # print("SelectFdr features")
    # print(featureScores.nlargest(10, 'Score'))  # print 10 best features

    feat_importances = pd.Series(fit.scores_, index=X_raw.columns).nlargest(20)
    feat_list = [feat_importances.index[i] for i in range(0, len(feat_importances))]
    feat_list_with_imp = [(str(feat_importances.index[i]), round(feat_importances[i], 7)) for i in
                          range(0, len(feat_importances))]
    # print(feat_list)

    # print(fit)
    # print(y_data)
    # print(x_new)
    # transformer = GenericUnivariateSelect(f_classif, 'k_best', param=20)
    # fit = transformer.fit(x_data, y_data)

    X = X_raw[feat_list]
    y = y_data
    # X_test = X_test[feat_list]
    X_test = []

    # feat_importances.plot(kind='barh')
    # plt.show()
    print("Completed SelectFdr")
    return X, y, X_test, ', '.join(feat_list), feat_list_with_imp


def feature_selectkbest_f_classif(X_train, X_raw, y_data):
    print("Running selectkbest")
    bestfeatures = SelectKBest(score_func=f_classif, k=10)
    fit = bestfeatures.fit(X_train, y_data)
    x_new = bestfeatures.transform(X_train)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X_raw.columns)
    # concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    # print("SelectPercentile features")
    # print(featureScores.nlargest(10, 'Score'))  # print 10 best features

    # print(fit)
    # print(y_data)
    # print(x_new)
    # transformer = GenericUnivariateSelect(f_classif, 'k_best', param=20)
    # fit = transformer.fit(x_data, y_data)

    feat_importances = pd.Series(fit.scores_, index=X_raw.columns).nlargest(20)
    feat_list = [feat_importances.index[i] for i in range(0, len(feat_importances))]
    feat_list_with_imp = [(str(feat_importances.index[i]), round(feat_importances[i], 7)) for i in
                          range(0, len(feat_importances))]
    # print(feat_list)

    X = X_raw[feat_list]
    y = y_data
    # X_test = X_test[feat_list]
    X_test = []

    print("Completed selectkbest")
    return X, y, X_test, ', '.join(feat_list), feat_list_with_imp


def feature_selectkbest_chi2(X_train, X_raw, y_data):
    print("Running selectkbest chi2")
    bestfeatures = SelectKBest(score_func=chi2, k=10)
    fit = bestfeatures.fit(X_train, y_data)
    x_new = bestfeatures.transform(X_train)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X_raw.columns)
    # concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    # print("SelectPercentile features")
    # print(featureScores.nlargest(10, 'Score'))  # print 10 best features

    # print(fit)
    # print(y_data)
    # print(x_new)
    # transformer = GenericUnivariateSelect(f_classif, 'k_best', param=20)
    # fit = transformer.fit(x_data, y_data)

    feat_importances = pd.Series(fit.scores_, index=X_raw.columns).nlargest(20)
    feat_list = [feat_importances.index[i] for i in range(0, len(feat_importances))]
    feat_list_with_imp = [(str(feat_importances.index[i]), round(feat_importances[i], 7)) for i in
                          range(0, len(feat_importances))]
    # print(feat_list)

    X = X_raw[feat_list]
    y = y_data
    # X_test = X_test[feat_list]
    X_test = []

    print("Completed selectkbest chi2")
    return X, y, X_test, ', '.join(feat_list), feat_list_with_imp


def feature_GenericUnivariateSelect(X_train, X_raw, y_data):
    print("Running GenericUnivariate")
    bestfeatures = GenericUnivariateSelect(f_classif, 'k_best', param=20)
    fit = bestfeatures.fit(X_train, y_data)
    x_new = bestfeatures.transform(X_train)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X_raw.columns)
    # concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    # print("GenericUnivariate features")
    # print(featureScores.nlargest(10, 'Score'))  # print 10 best features

    # print(fit)
    # print(y_data)
    # print(x_new)
    # transformer = GenericUnivariateSelect(f_classif, 'k_best', param=20)
    # fit = transformer.fit(x_data, y_data)

    feat_importances = pd.Series(fit.scores_, index=X_raw.columns).nlargest(20)
    feat_list = [feat_importances.index[i] for i in range(0, len(feat_importances))]
    feat_list_with_imp = [(str(feat_importances.index[i]), round(feat_importances[i], 7)) for i in
                          range(0, len(feat_importances))]
    # print(feat_list)

    X = X_raw[feat_list]
    y = y_data
    # X_test = X_test[feat_list]
    X_test = []

    print("Completed GenericUnivariate")
    return X, y, X_test, ', '.join(feat_list), feat_list_with_imp



def feature_SelectPercentile(X_train, X_raw, y_data):
    print("Running SelectPercentile")
    bestfeatures = SelectPercentile(f_classif, percentile=10)
    fit = bestfeatures.fit(X_train, y_data)
    x_new = bestfeatures.transform(X_train)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X_raw.columns)
    # concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    # print("SelectPercentile features")
    # print(featureScores.nlargest(10, 'Score'))  # print 10 best features

    # print(fit)
    # print(y_data)
    # print(x_new)
    # transformer = GenericUnivariateSelect(f_classif, 'k_best', param=20)
    # fit = transformer.fit(x_data, y_data)

    feat_importances = pd.Series(fit.scores_, index=X_raw.columns).nlargest(20)
    feat_list = [feat_importances.index[i] for i in range(0, len(feat_importances))]
    feat_list_with_imp = [(str(feat_importances.index[i]), round(feat_importances[i], 7)) for i in
                          range(0, len(feat_importances))]
    # print(feat_list)

    X = X_raw[feat_list]
    y = y_data
    # X_test = X_test[feat_list]
    X_test = []

    print("Completed SelectPercentile")
    return X, y, X_test, ', '.join(feat_list), feat_list_with_imp


def feature_RFE(X_train, X_raw, y_data):
    print("Running RFE")
    estimator = SVR(kernel="linear")
    bestfeatures = RFE(estimator, 5, step=1)
    fit = bestfeatures.fit(X_train, y_data)
    x_new = bestfeatures.transform(X_train)
    dfscores = pd.DataFrame(fit.ranking_)
    dfcolumns = pd.DataFrame(X_raw.columns)
    # concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    # print("RFE features")
    # print(featureScores.nlargest(10, 'Score'))  # print 10 best features

    # print(fit)
    # print(y_data)
    # print(x_new)
    # transformer = GenericUnivariateSelect(f_classif, 'k_best', param=20)
    # fit = transformer.fit(x_data, y_data)

    feat_importances = pd.Series(fit.ranking_, index=X_raw.columns).nlargest(20)
    # print(feat_importances)
    feat_list = [feat_importances.index[i] for i in range(0, len(feat_importances))]
    # print(feat_list)

    X = X_raw[feat_list]
    y = y_data
    # X_test = X_test[feat_list]
    X_test = []

    print("Completed RFE")
    return X, y, X_test, ', '.join(feat_list)



def feature_RFECV(X_train, X_raw, y_data):
    print("Running RFECV")
    estimator = SVR(kernel="linear")
    bestfeatures = RFECV(estimator, step=1, cv=5)
    fit = bestfeatures.fit(X_train, y_data)
    x_new = bestfeatures.transform(X_train)
    dfscores = pd.DataFrame(fit.ranking_)
    dfcolumns = pd.DataFrame(X_raw.columns)
    # concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    # print("RFECV features")
    # print(featureScores.nlargest(10, 'Score'))  # print 10 best features

    # print(fit)
    # print(y_data)
    # print(x_new)
    # transformer = GenericUnivariateSelect(f_classif, 'k_best', param=20)
    # fit = transformer.fit(x_data, y_data)

    feat_importances = pd.Series(fit.ranking_, index=X_raw.columns).nlargest(20)
    feat_list = [feat_importances.index[i] for i in range(0, len(feat_importances))]
    # print(feat_list)

    X = X_raw[feat_list]
    y = y_data
    # X_test = X_test[feat_list]
    X_test = []

    print("Completed RFECV")
    return X, y, X_test, ', '.join(feat_list)


def feature_VarianceThreshold(X_train, X_raw, y_data):
    print("Running VarianceThreshold")
    # x_data, y_data, test_dataset = readDataFromFile()
    bestfeatures = VarianceThreshold(threshold=(.8 * (1 - .8)))
    fit = bestfeatures.fit(X_train, y_data)
    x_new = bestfeatures.transform(X_train)
    dfcolumns = pd.DataFrame(X_raw.columns)
    # concat two dataframes for better visualization
    # print("SelectPercentile features")
    # print(featureScores.nlargest(10, 'Score'))  # print 10 best features

    # print(fit)
    # print(y_data)
    # print(x_new)
    # transformer = GenericUnivariateSelect(f_classif, 'k_best', param=20)
    # fit = transformer.fit(x_data, y_data)

    feat_importances = pd.Series(fit.get_support(), index=X_raw.columns)
    feat_list = []

    for col in X_raw.columns:
        if feat_importances[col]:
            feat_list.append(col)

    # print(feat_list)

    X = X_raw[feat_list]
    y = y_data
    # X_test = X_test[feat_list]
    X_test = []

    print("Completed VarianceThreshold")
    return X, y, X_test, ', '.join(feat_list)

# from pyswarms.utils.functions import single_obj as fx
# def feature_GlobalBestPSO():
#     X, y, test_dataset = readDataFromFile()
#     # print(X.head())
#     # Set-up hyperparameters
#     options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
#
#     # Call instance of PSO
#     optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=190, options=options)
#
#     print(type(fx.sphere))
#     # Perform optimization
#     # cost, pos = optimizer.optimize(X, print_step=100, iters=1000, verbose=3)

from sklearn.feature_extraction import DictVectorizer

def feature_mutual_info(X_train, X_raw, y_data):
    print("Running mutual_info")
    # X, y, test_dataset = readDataFromFile()
    # x_data = X
    # y_data = y

    feature_scores = mutual_info_classif(X_train, y_data)
    feat_importances = pd.Series(feature_scores, index=X_raw.columns).nlargest(20)
    # print(feat_importances)
    feat_list = [feat_importances.index[i] for i in range(0, len(feat_importances))]
    feat_list_with_imp = [(str(feat_importances.index[i]), round(feat_importances[i], 7)) for i in
                          range(0, len(feat_importances))]
    # print(feat_list_with_imp)
    # print(feature_scores)
    # print(feat_list)
    X = X_raw[feat_list]
    y = y_data
    # X_test = X_test[feat_list]
    X_test = []

    print("Completed mutual_info")
    return X, y, X_test, ', '.join(feat_list), feat_list_with_imp


# ('idx:', 150, ' score:', 0.006270470912909776)
# ('idx:', 153, ' score:', 0.006270470912909776)
# ('idx:', 154, ' score:', 0.006270470912909776)
# ('idx:', 166, ' score:', 0.006270470912909776)
# ('idx:', 170, ' score:', 0.006270470912909776)
# ('idx:', 174, ' score:', 0.006270470912909776)
# ('idx:', 178, ' score:', 0.006270470912909776)
# ('idx:', 0, ' score:', 0.006270470912908888)
# ('idx:', 149, ' score:', 0.006270470912908888)
# Scores for conditional_mutual_info is all same so can just go by count
def feature_conditional_mutual_info_maximisation(X_train, X_raw, y_data):
    print("Running conditional_mutual_info")
    features_scores = CMIM.cmim(X_train, y_data, n_selected_features=20)
    features_index = [int(index[0]) for index in features_scores]
    feat_list = X_raw.columns.values[features_index]
    feat_list_with_imp = [(feat_list[i], features_scores[i][1]) for i in range(len(features_scores))]
    # print("features_index:", features_index)
    # print("feat_list_with_imp:", feat_list_with_imp)
    X = X_raw[feat_list]
    y = y_data
    # X_test = X_test[feat_list]
    X_test = []

    print("Completed conditional_mutual_info")
    return X, y, X_test, ', '.join(feat_list), feat_list_with_imp


def feature_joint_mutual_info(X_train, X_raw, y_data):
    print("Running joint_mutual_info")
    features_scores = JMI.jmi(X_train, y_data, n_selected_features=20)

    features_index = [int(index[0]) for index in features_scores]
    # print("features_index:", features_index)
    feat_list = X_raw.columns.values[features_index]
    feat_list_with_imp = [(feat_list[i], features_scores[i][1]) for i in range(len(features_scores))]
    # print("feat_list_with_imp:", feat_list_with_imp)
    X = X_raw[feat_list]
    y = y_data
    # X_test = X_test[feat_list]
    X_test = []

    print("Completed joint_mutual_info")
    return X, y, X_test, ', '.join(feat_list), feat_list_with_imp


def feature_mutual_info_maximisation(X_train, X_raw, y_data):
    print("Running mutual_info_maximisation")
    features_scores = MIM.mim(X_train, y_data, n_selected_features=20)

    features_index = [int(index[0]) for index in features_scores]
    # print("features_index:", features_index)
    feat_list = X_raw.columns.values[features_index]
    feat_list_with_imp = [(feat_list[i], features_scores[i][1]) for i in range(len(features_scores))]
    # print("feat_list_with_imp:", feat_list_with_imp)
    X = X_raw[feat_list]
    y = y_data
    # X_test = X_test[feat_list]
    X_test = []

    print("Completed mutual_info_maximisation")
    return X, y, X_test, ', '.join(feat_list), feat_list_with_imp


def feature_max_relevance_min_redundancy(X_train, X_raw, y_data):
    print("Running max_relevance_min_redundancy")
    features_scores = MRMR.mrmr(X_train, y_data, n_selected_features=20)

    features_index = [int(index[0]) for index in features_scores]
    # print("features_index:", features_index)
    feat_list = X_raw.columns.values[features_index]
    feat_list_with_imp = [(feat_list[i], features_scores[i][1]) for i in range(len(features_scores))]
    # print("feat_list_with_imp:", feat_list_with_imp)
    X = X_raw[feat_list]
    y = y_data
    # X_test = X_test[feat_list]
    X_test = []

    print("Completed max_relevance_min_redundancy")
    return X, y, X_test, ', '.join(feat_list), feat_list_with_imp


def ridge_regression(alpha, X_train, y_train, X_test, y_test=None):
    #Fit the model
    ridgereg = Ridge(alpha=alpha,normalize=True)
    ridgereg.fit(X_train, y_train)
    y_pred = ridgereg.predict(X_test)

    evaluation_metrics(y_test, y_pred)


def best_fit_ridge_regression():
    alpha_ridge = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]

    X, y, temp = readDataFromFile()
    # print(X.head())

    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.25, random_state=21)
    # Feature Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    for i in range(10):
        print("ridge_regression for alpha:"+ str(alpha_ridge[i]))
        ridge_regression(alpha_ridge[i], X_train, y_train, X_test, y_test)
        print("\n")


def lasso_regression(alpha, X_train, y_train, X_test, y_test=None):
    #Fit the model
    lassoreg = Lasso(alpha=alpha, normalize=True, max_iter=1e5)
    lassoreg.fit(X_train, y_train)
    y_pred = lassoreg.predict(X_test)

    evaluation_metrics(y_test, y_pred)


def best_fit_lasso_regression():
    print("Running lasso_regression")
    alpha_lasso = [1e-15, 1e-10, 1e-8, 1e-5, 1e-4, 1e-3, 1e-2, 1, 5, 10]
    X, y, temp = readDataFromFile()
    # print(X.head())

    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.25, random_state=21)
    # Feature Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    for i in range(10):
        print("lasso_regression for alpha:"+ str(alpha_lasso[i]))
        lasso_regression(alpha_lasso[i], X_train, y_train, X_test, y_test)
        print("\n")


def recursive_feature_elimination():
    print("Running recursive_feature_elimination")
    X, y, temp = readDataFromFile()

    # svm = LinearSVC()
    svr =SVR(kernel="linear")
    # Feature Scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # mlp = MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=500)
    # create the RFE model for the svm classifier
    # and select attributes
    rfe = RFE(svr, 100)
    rfe = rfe.fit(X, y)
    # print summaries for the selection of attributes
    print("rfe.support_:", rfe.support_)
    print("rfe.ranking_:", rfe.ranking_)



def kmeans_clustering():
    X, y, temp = readDataFromFile()
    print(X.head())
    # print(X.iloc[:,0])
    for col in X.columns.values:

        # df = pd.DataFrame({
        #     'X': X[col],
        #     'y': y
        # })
        df = pd.DataFrame({
            'X': X[col]
        })

        kmeans = KMeans(n_clusters=2)
        kmeans.fit(df)

        # labels = kmeans.predict(df)
        centroids = kmeans.cluster_centers_
        fig = plt.figure(figsize=(5, 5))
        # colmap = {1: 'r', 2: 'g', 3: 'b'}
        colmap = {1: 'r', 2: 'g'}
        colors = map(lambda x: colmap[x + 1], kmeans.labels_)

        plt.scatter(df['X'], df['y'], c=kmeans.labels_, cmap='rainbow')
        # plt.scatter(df['X'], c=kmeans.labels_, cmap='rainbow')
        # plt.plot(df['X'])
        # for idx, centroid in enumerate(centroids):
        #     plt.scatter(*centroid, color=colmap[idx + 1])
        # plt.xlim(0, 80)
        # plt.ylim(0, 80)
        plt.suptitle(col)
        plt.show()

def readDtaFile():
    data = pd.read_stata("child_growth_dataset/NHMRC.dta")
    print(data.head())
    data.to_csv("NHMRC_we_generated.csv")
    # for chunk in data:
    #     print(chunk)

    return ''

def binary_pso():
    # Create an instance of the classifier
    classifier = linear_model.LogisticRegression()

    # Define objective function
    def f_per_particle(m, alpha):
        """Computes for the objective function per particle

        Inputs
        ------
        m : numpy.ndarray
            Binary mask that can be obtained from BinaryPSO, will
            be used to mask features.
        alpha: float (default is 0.5)
            Constant weight for trading-off classifier performance
            and number of features

        Returns
        -------
        numpy.ndarray
            Computed objective function
        """
        total_features = 15
        # Get the subset of the features from the binary mask
        if np.count_nonzero(m) == 0:
            X_subset = X
        else:
            X_subset = X[:, m == 1]
        # Perform classification and store performance in P
        classifier.fit(X_subset, y)
        P = (classifier.predict(X_subset) == y).mean()
        # Compute for the objective function
        j = (alpha * (1.0 - P)
             + (1.0 - alpha) * (1 - (X_subset.shape[1] / total_features)))

        return j


def f(x, alpha=0.88):
    """Higher-level method to do classification in the
    whole swarm.

    Inputs
    ------
    x: numpy.ndarray of shape (n_particles, dimensions)
        The swarm that will perform the search

    Returns
    -------
    numpy.ndarray of shape (n_particles, )
        The computed loss for each particle
    """
    n_particles = x.shape[0]
    j = [f_per_particle(x[i], alpha) for i in range(n_particles)]
    return np.array(j)




# print("Running L1_based_feature_selection")
#     X, y, temp = readDataFromFile()
#     print(X.shape)
#     print(X.columns.values.tolist())
#
#     alpha_lasso = [1e-15, 1e-10, 1e-8, 1e-5, 1e-4, 1e-3, 1e-2, 1, 5, 10]
#     for i in range(10):
#         print("L1_based_feature_selection for alpha:"+ str(alpha_lasso[i]))
#         lsvc = LinearSVC(C=alpha_lasso[i], penalty="l1", dual=False).fit(X, y)
#         model = SelectFromModel(lsvc, prefit=True)
#         X_new = model.transform(X)
#         print(X_new.shape)
#         print(X.columns.values.tolist())
#         X_train, X_test, y_train, y_test = train_test_split(X_new.values, y.values, test_size=0.25, random_state=21)
#         # Feature Scaling
#         scaler = StandardScaler()
#         X_train = scaler.fit_transform(X_train)
#         X_test = scaler.transform(X_test)
#
#         multi_layer_perceptron_classifier(X_train, y_train, X_test, test_dataset, y_test, cross_validation=True)
#
#         print("\n")

def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],4)
        self.weights2   = np.random.rand(4,1)
        self.y          = y
        self.output     = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2


# if __name__ == "__main__":
#     X = np.array([[0,0,1],
#                   [0,1,1],
#                   [1,0,1],
#                   [1,1,1]])
#     y = np.array([[0],[1],[1],[0]])
#
#     train_dataset = pd.read_csv('train_dataset.csv')
#     test_dataset = pd.read_csv('test_dataset.csv')
#
#     X = train_dataset.iloc[:, 2:len(train_dataset.columns) - 1].values
#     y = train_dataset.iloc[:, len(train_dataset.columns) - 1].values
#     nn = NeuralNetwork(X,y)

    # for i in range(10):
    #     nn.feedforward()
    #     nn.backprop()

    # print(nn.output[:10])


# get_all_classifier_result()
# best_fit_ridge_regression()
# best_fit_lasso_regression()
# recursive_feature_elimination()
# L1_based_feature_selection([], [])
# crossValidation()

# feature_GlobalBestPSO()
# get_all_classifier_result()


# FeatureImportance('', '')
# feature_SelectFpr('', '')

# X, y, temp = readDataFromFile()
# feature_RFE(X, y)
# feature_conditional_mutual_info_maximisation(X, y)
# feature_joint_mutual_info(X, y)
# feature_mutual_info_maximisation(X, y)
# feature_max_relevance_min_redundancy(X, y)

# feature_mutual_info('', '')
# results_plot()

# line_graph()
# pickle_in = "10_model_cumulative_k_fold_results_knn.pickle"
# pickle_in = "100_model_cumulative_k_fold_results_knn.pickle"
# pickle_in = "10_model_cumulative_k_fold_results_non_imputed.pickle"
pickle_in = "10_model_cumulative_k_fold_results_non_imputed_test_columns.pickle"
# pickle_in = "2_model_cumulative_k_fold_results_Breast_Cancer_Wisconsin.pickle"

# line_graph_model(pickle_in)
line_graph_model(pickle_in, is_accuracy=True)
# feat_grt_5_top_5_mean(pickle_in, is_accuracy=True)
# feature_VarianceThreshold([],[])
# temp_plot()


# plot_top_5_mean_50_fold_classif_result()


n_splits = 5
n_repeats = 10
pickle_filename = str(n_splits*n_repeats)+"_repeated_k_fold_results_non_imputed_test_columns.pickle"
# pickle_filename = str(n_splits*n_repeats)+"_repeated_k_fold_results_non_imputed.pickle"
# # pickle_filename = str(n_splits*n_repeats)+"_repeated_k_fold_results_knn_mode.pickle"
# # pickle_filename = str(n_splits*n_repeats)+"_repeated_k_fold_results_Breast_Cancer_Wisconsin.pickle"
perform_repeated_k_fold(n_splits, n_repeats, pickle_filename)


#pickle_in = open(pickle_filename, "rb")
#results_dict = pickle.load(pickle_in)
# print(results_dict)
#for key, value in results_dict.items():
#   if key == "all" or 'feat_list' not in value:
#        continue
#   value['feat_list'].sort(key=lambda item:item[1], reverse=True)
#   print("KEY:",key)
#   print("features:", [k for k,v in value['feat_list']][:10])


n_splits = 5
n_repeats = 10
write_pickle_filename = str(n_splits*n_repeats)+"_model_cumulative_k_fold_results_non_imputed_test_columns.pickle"
part_pickle_filename = "_model_cumulative_k_fold_results_non_imputed_test_columns.pickle"
# write_pickle_filename = str(n_splits*n_repeats)+"_model_cumulative_k_fold_results_non_imputed.pickle"
# part_pickle_filename = "_model_cumulative_k_fold_results_non_imputed.pickle"
# write_pickle_filename = str(n_splits*n_repeats)+"_model_cumulative_k_fold_results_weighted_knn.pickle"
# part_pickle_filename = "_model_cumulative_k_fold_results_weighted_knn.pickle"
# write_pickle_filename = str(n_splits*n_repeats)+"_model_cumulative_k_fold_results_knn.pickle"
# part_pickle_filename = "_model_cumulative_k_fold_results_knn.pickle"
# write_pickle_filename = str(n_splits*n_repeats)+"_model_cumulative_k_fold_results_Breast_Cancer_Wisconsin.pickle"
# part_pickle_filename = "_model_cumulative_k_fold_results_Breast_Cancer_Wisconsin.pickle"
perform_repeated_k_fold_on_model(n_splits, n_repeats, pickle_filename, write_pickle_filename, part_pickle_filename)


# violin_swarm_plot()
# feat_list = ["infweight_6month", "infmuac_6month", "mweight_32week", "mheight", "fheight", "mweight_enroll", "fweight", "mweight_6month"]
# heat_map(feat_list)

# kmeans_clustering()

# readDtaFile()

# mlp_feat_list = ["infweight_6month", "infmuac_6month", "mweight_32week", "mheight", "fheight", "mweight_enroll", "fweight", "mweight_6month"]
mlp_feat_list = ['infweight_6month', 'mheight', 'infhead_6month', 'fheight', 'fweight', 'mweight_6month']
# 'infweight_6month, infmuac_6month, mheight, infhead_6month, fheight, mweight_enroll, fweight'
# svm_feat_list = ["infweight_6month", "mheight"]
svm_feat_list = ['mweight_32week', 'mvitd_32week', 'bftime', 'infweight_6month', 'mb12_32week', 'mb12_enroll']
# rndf_feat_list = ["infweight_6month", "mheight", "fheight"]
rndf_feat_list = ['infweight_6month', 'mweight_32week', 'mheight', 'infweight_birth', 'infhead_6month', 'fheight', 'mweight_enroll', 'fweight', 'mweight_6month']

# fold_50_t_test(mlp_feat_list, svm_feat_list, rndf_feat_list, accurracy=True)


# actual_test_set_result(mlp_feat_list, svm_feat_list, rndf_feat_list)

# mlp_svm_results: Ttest_indResult(statistic=7.423254164661326, pvalue=4.2203190903502e-11)  - statistic= 7.423254165 , pvalue= 0.0
# mlp_rndf_results: Ttest_indResult(statistic=1.4923087391104006, pvalue=0.13883054102834777)   - statistic= 1.492308739 , pvalue= 0.138830541
# rndf_svm_results: Ttest_indResult(statistic=6.3819722281489994, pvalue=5.824348989739631e-09)   - statistic= 6.381972228 , pvalue= 6e-09

#REMEBER TO REMOVE DUMMY COLUMNS(id, unnamed) FROM DATASET AS EVERYTHING IS USED FOR PROCESS, SEARCH AND DELETE THE COLUMNS MANUALLY









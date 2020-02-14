import pandas as pd
import random
import math
import numpy as np
from itertools import combinations
from sklearn.neighbors.kde import KernelDensity

from sklearn.preprocessing import LabelEncoder

from fancyimpute import KNN
from fancyimpute import IterativeImputer


def main_fun():
    # imputation()
    lt_60_percent_non_missing_data()


def lt_60_percent_non_missing_data():
    data = pd.read_csv('NHMRC.csv')
    print(data)
    all_dropped_data = data.dropna()
    print(all_dropped_data)
    total_columns = data.columns.values
    len(total_columns)
    print(len(total_columns))
    data = get_lt_percent_missing_data(data, 60)
    print(len(data.columns.values))
    dropped_data = data.dropna()
    print(dropped_data)
    # dropped_data.to_csv('non_imputed_full_test_data.csv')

def imputation():
    data = pd.read_csv('NHMRC.csv')
    # data = pd.read_csv('NHMRC_test.csv')
    data = get_lt_percent_missing_data(data, 60)
    non_imputed_full_train_data = data.dropna()
    non_imputed_full_train_data.to_csv('non_imputed_full_data.csv')
    knn_mode_imputed_data = fill_disc_cont_separately(data, 'weightedknn', 'knn')
    knn_mode_imputed_data.to_csv('knn_mode_imputed_data.csv')
    test_data = pd.read_csv('NHMRC_test.csv')
    test_data = test_data.dropna()
    test_data.to_csv('non_imputed_full_test_data.csv')



def fill_disc_cont_separately(data, disc_method, cont_method):
    discrete_data, continuous_data = get_cont_disc_data(data)
    filled_discrete_data = fill_discrete_data(discrete_data, disc_method)
    filled_continuous_data = fill_cont_data(continuous_data, cont_method)
    filled_discrete_data.to_csv('filled_discrete_data.csv')
    filled_continuous_data.to_csv('filled_continuous_data.csv')
    merged_data = pd.concat([filled_discrete_data, filled_continuous_data], axis=1)
    return merged_data


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


def fill_discrete_data(data, method):
    if method == 'mode':
        return mode_discrete_impute(data)
    elif method == 'mice':
        return mice_impute(data)
    elif method == 'knn':
        return knn_impute(data, 5)
    elif method == 'weightedknn':
        return weighted_knn_imputation(data, 11)
    else:
        print("method is not defined")
        quit(0)


def fill_cont_data(data, method):
    if method == 'knn':
        return knn_impute(data, 5)
    elif method == 'mice':
        return mice_impute(data)
    elif method == 'std':
        return std_impute(data)
    else:
        print("method is not defined")
        quit(0)


def weighted_knn_imputation(data, k):
    attribute_weight_datframe = pd.DataFrame(columns=data.columns)
    print(attribute_weight_datframe)
    copied_data = data.copy()
    print(data)
    print(copied_data)
    for row, rowitem in data.iterrows():
        for column, columnitem in rowitem.iteritems():
            if columnitem != columnitem:
                print("row : ", row, "column : ", column)
                attribute_weight_datframe = attribute_weights(data, column, attribute_weight_datframe)
                # print(attribute_weight_datframe)
                copied_data.loc[row, column] = return_weighted_knn_imputed_value(data, row, column, attribute_weight_datframe, k)
        # copied_data.to_csv('temporary_filled_data.csv')
    return copied_data


def attribute_weights(data, column, attribute_weight_datframe):
    if column in attribute_weight_datframe.index:
        return attribute_weight_datframe
    for c in data.columns:
        feature1 = column
        feature2 = c
        if feature1 == feature2:
            attribute_weight_datframe.loc[feature1, feature2] = 0
            continue
        column1_unique = data[feature1].value_counts()
        column2_unique = data[feature2].value_counts()
        data1 = data[[feature1, feature2]]
        combination_count = data1.groupby([feature1, feature2]).size().reset_index(name='Count')
        attribute1_size = np.size(column1_unique)
        attribute2_size = np.size(column2_unique)
        n = combination_count['Count'].sum()
        total_weight = 0
        for row, rowitem in combination_count.iterrows():
            term2 = (column1_unique[rowitem[feature1]] * column2_unique[rowitem[feature2]]) / n
            cuurent_weight = math.pow((rowitem['Count'] - term2), 2) / term2
            total_weight = total_weight + cuurent_weight
        total_weight = total_weight / n
        total_weight = math.sqrt(total_weight / min(attribute1_size, attribute2_size))
        attribute_weight_datframe.loc[feature1, feature2] = total_weight
    return attribute_weight_datframe


def return_weighted_knn_imputed_value(data, row, column, attribute_weight_datframe, k):
    index_distance_neighbour_weights_list = nearest_neighbour_distance_weight(data, data.loc[row], attribute_weight_datframe, column, k + 1)
    categorical_data_neighbour_weights_list = {}
    for index, distance, kernel_weight in index_distance_neighbour_weights_list:
        # print(data.loc[index, column])
        cat_value = data.loc[index, column]
        categorical_data_neighbour_weights_list[
            cat_value] = kernel_weight if cat_value not in categorical_data_neighbour_weights_list else \
            categorical_data_neighbour_weights_list[cat_value] + kernel_weight
    categorical_data_neighbour_weights_list = {key: val for key, val in categorical_data_neighbour_weights_list.items()
                                               if key == key}
    v = list(categorical_data_neighbour_weights_list.values())
    k = list(categorical_data_neighbour_weights_list.keys())
    cat_value_to_impute = k[v.index(max(v))]
    print("imputed value : ", cat_value_to_impute)
    return cat_value_to_impute


def nearest_neighbour_distance_weight(data, input_row, attribute_weight_dataframe, column, k):
    distance_index_list = []
    for rowindex, rowitem in data.iterrows():
        distance_index_list.append((weighted_distance(input_row, rowitem, column, attribute_weight_dataframe), rowindex))
    distance_list = [value[0] for value in distance_index_list]
    distance_array = np.array(distance_list).reshape(-1, 1)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(distance_array)
    distance_kernel_scores = kde.score_samples(distance_array)
    distance_index_list.sort()
    nearest_neighbour_distances = distance_index_list[1:k]
    index_distance_kernel_weight_list = []
    for distance, index in nearest_neighbour_distances:
        index_distance_kernel_weight_list.append((index, distance, distance_kernel_scores[index]))
    kernel_weights_sum = sum([value[2] for value in index_distance_kernel_weight_list])
    index_distance_neighbour_weights_list = []
    for index, distance, kernel_weight in index_distance_kernel_weight_list:
        index_distance_neighbour_weights_list.append((index, distance, kernel_weight / kernel_weights_sum))
    return index_distance_neighbour_weights_list


def weighted_distance(row1, row2, inputcolumn, attribute_weight_datframe):
    # print(row1)
    # print(row2)
    distance = 0
    number_of_non_nan_columns = 0
    for column, columnitem in row1.iteritems():
        # print(row1.loc[column])
        # print(row2.loc[column])
        if not (is_nan_value(row1.loc[column]) or is_nan_value(row2.loc[column])):
            number_of_non_nan_columns = number_of_non_nan_columns + 1
            if row1.loc[column] != row2.loc[column]:
                distance = distance + (2 * attribute_weight_datframe.loc[inputcolumn,column])
                # print(number_of_non_nan_columns)
                # print(distance)
    # print(distance)
    # print(number_of_non_nan_columns)
    distance = math.sqrt(distance / number_of_non_nan_columns)
    return distance
    # print(distance)


def mode_discrete_impute(data):
    for column in data.columns.tolist():
        data[column].fillna((data[column].mode()[0]), inplace=True)
    return data


def std_impute(data):
    for column in data.columns.tolist():
        print("imputing continuous data using std")
        column_name_avg = data[column].mean()
        column_name_std = data[column].std()
        min_value = column_name_avg - column_name_std
        max_value = column_name_avg + column_name_std
        data[column].fillna((random.uniform(min_value, max_value)), inplace=True)
    return data


def knn_impute(data, n):
    print("imputing data using knn")
    data_matrix = data.values
    filled_data = pd.DataFrame(KNN(n).fit_transform(data_matrix))
    filled_data.columns = data.columns
    filled_data.index = data.index
    filled_data.to_csv('knn_imputed_data.csv')
    print("data imputed using knn")
    return filled_data


def mice_impute(data):
    print("imputing data using mice")
    data_matrix = data.values
    filled_data = pd.DataFrame(
        IterativeImputer(imputation_order='random', n_iter=5, sample_posterior=True).fit_transform(data_matrix))
    filled_data.columns = data.columns
    filled_data.index = data.index
    filled_data.to_csv('mice_imputed_data.csv')
    print("data imputed using mice")


def get_lt_percent_missing_data(data, percent):
    data_column_percent_null_values = pd.DataFrame((data.isnull().sum() / data.isnull().count()) * 100)
    lt_60_percent_columnlist = data_column_percent_null_values[data_column_percent_null_values.iloc[:, 0]
                                                               < percent].index.tolist()
    gt_60_percent_columnlist = data_column_percent_null_values[data_column_percent_null_values.iloc[:, 0]
                                                               >= percent].index.tolist()
    print(gt_60_percent_columnlist)

    # null_stunting36_index = []
    # print(data.shape)
    # for index, data_row in data.iterrows():
    #     if pd.isna(data_row['stunting36']):
    #         null_stunting36_index.append(index)
    # data.drop(null_stunting36_index, axis=0, inplace=True)
    # print(data.shape)
    return get_data_from_column_list(data, lt_60_percent_columnlist)


def get_cont_disc_data(data):
    categorical_feature_mask = data.dtypes == object
    categorical_cols = data.columns[categorical_feature_mask].tolist()
    continuous_data = data.drop(categorical_cols, axis=1)
    discrete_data = data[categorical_cols]
    discrete_data.to_csv('discrete_data.csv')
    continuous_data.to_csv('continuous_data.csv')
    return discrete_data, continuous_data


def get_column_list_null_values(data, percent):
    data_column_percent_null_values = pd.DataFrame((data.isnull().sum() / data.isnull().count()) * 100)
    return data_column_percent_null_values[data_column_percent_null_values.iloc[:, 0] > percent].index.tolist()


def drop_data_from_column_list(data, list):
    return data.drop(list, axis=1)


def get_data_from_column_list(data, list):
    return data[list]


def tarun_convert_decimal_data(raw_data):
    # Tarun work one-hot encoding
    print(raw_data.columns.values)
    cols_with_36 = []
    for col in raw_data.columns.values:
        if "36" in col:
            cols_with_36.append(col)
    cols_with_36.remove("stunting36")
    print(cols_with_36)

    cols_drop = ['mage_cat', 'mheight_cat', 'mbmi_cat_enroll', 'fbmi_cat', 'preterm', 'mbmi_6month_cat',
                 'finalgestdel_cat', 'HAZ36']

    # cols_drop = ['mage_cat', 'mheight_cat', 'mbmi_cat_enroll', 'fbmi_cat', 'preterm', 'mbmi_6month_cat', 'HAZ36',
    #              'HAZ6', 'HAZ6w', 'stunting6', 'stunting6w']

    raw_data.drop(cols_drop + cols_with_36, axis=1, inplace=True)
    # return ""

    # cols_index = [raw_data.columns.get_loc(col) for col in ['mage_cat', 'mheight_cat', 'mbmi_cat_enroll', 'fbmi_cat', 'preterm', 'mbmi_6month_cat']]
    # raw_data.drop(raw_data.columns[cols_index], axis=1, inplace=True)
    # raw_data.drop(['mage_cat'], axis=1)
    # print(raw_data['mage_cat'].head())

    raw_data = pd.get_dummies(raw_data,
                              columns=['arm', 'mjob', 'medu', 'pregnum_cat', 'medu_cat', 'mjob_cat', 'iron_dose',
                                       'finalgestdel_cat', 'season', 'deltype', 'wi_quint', 'fjob', 'fedu', 'fedu_cat',
                                       'fjob_cat'],
                              prefix=['arm', 'mjob', 'medu', 'pregnum_cat', 'medu_cat', 'mjob_cat', 'iron_dose',
                                      'finalgestdel_cat', 'season', 'deltype', 'wi_quint', 'fjob', 'fedu', 'fedu_cat',
                                      'fjob_cat'])
    replace_map = {'presupp': {'yes': 1, 'no': 0},
                   'suppfe': {'yes': 1, 'no': 0},
                   'suppvitb': {'yes': 1, 'no': 0},
                   'suppcalc': {'yes': 1, 'no': 0},
                   'suppvita': {'yes': 1, 'no': 0},
                   'suppvitc': {'yes': 1, 'no': 0},
                   'suppvite': {'yes': 1, 'no': 0},
                   'suppmmn': {'yes': 1, 'no': 0},
                   'supptrad': {'yes': 1, 'no': 0},
                   'dietchng': {'yes': 1, 'no': 0},
                   'nblind': {'yes': 1, 'no': 0},
                   'epds_enroll': {'yes': 1, 'no': 0},
                   'mfolate_enroll_low': {'yes': 1, 'no': 0},
                   'mb12_enroll_low': {'yes': 1, 'no': 0},
                   'mhb_enroll_low': {'yes': 1, 'no': 0},
                   'mferritin_enroll_low': {'yes': 1, 'no': 0},
                   'mhb_32week_low': {'yes': 1, 'no': 0},
                   'pregnorm': {'yes': 1, 'no': 0},
                   'miodine_32week_low': {'yes': 1, 'no': 0},
                   'mferritin_32week_low': {'yes': 1, 'no': 0},
                   'mb12_32week_low': {'yes': 1, 'no': 0},
                   'mfolate_32week_low': {'yes': 1, 'no': 0},
                   'epds_32week': {'yes': 1, 'no': 0},
                   'infweight_birth_low': {'yes': 1, 'no': 0},
                   'milk_6week': {'yes': 1, 'no': 0},
                   'exclusive_6week': {'yes': 1, 'no': 0},
                   'formula_6week': {'yes': 1, 'no': 0},
                   'infvac_6week': {'yes': 1, 'no': 0},
                   'infsupp_6week': {'yes': 1, 'no': 0},
                   'infdiah_6week': {'yes': 1, 'no': 0},
                   'infcough_6week': {'yes': 1, 'no': 0},
                   'inffev_6week': {'yes': 1, 'no': 0},
                   'infhosp_6week': {'yes': 1, 'no': 0},
                   'bffeed': {'yes': 1, 'no': 0},
                   'milk_6month': {'yes': 1, 'no': 0},
                   'exclusive_6month': {'yes': 1, 'no': 0},
                   'formula_6month': {'yes': 1, 'no': 0},
                   'suppfood': {'yes': 1, 'no': 0},
                   'infvac_6month': {'yes': 1, 'no': 0},
                   'infsupp_6month': {'yes': 1, 'no': 0},
                   'infdiah_6month': {'yes': 1, 'no': 0},
                   'mhb_6month_low': {'yes': 1, 'no': 0},
                   'infhb_6month_low': {'yes': 1, 'no': 0},
                   'mferritin_6month_low': {'yes': 1, 'no': 0},
                   'stunting6': {'yes': 1, 'no': 0},
                   'epds_6month': {'yes': 1, 'no': 0},
                   'infcough_6month': {'yes': 1, 'no': 0},
                   'inffev_6month': {'yes': 1, 'no': 0},
                   'paint': {'yes': 1, 'no': 0},
                   'grdfloor': {'yes': 1, 'no': 0},
                   'highway': {'yes': 1, 'no': 0},
                   'wellwater': {'yes': 1, 'no': 0},
                   'pesticide': {'yes': 1, 'no': 0},
                   'mosqspay': {'yes': 1, 'no': 0},
                   'industrial': {'yes': 1, 'no': 0},
                   'stunting36': {'yes': 1, 'no': 0},
                   'stunting6w': {'yes': 1, 'no': 0},
                   'infsex': {'male': 1, 'female': 0},
                   }
    raw_data.replace(replace_map, inplace=True)
    raw_data.to_csv('tarun_converted_data.csv')
    return raw_data


def is_nan_value(value):
    return value != value


main_fun()

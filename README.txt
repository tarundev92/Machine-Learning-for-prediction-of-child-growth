The below functions assumes the train dataset to be in "NHMRC.csv" file and test dataset to be in "NHMRC_test.csv" file. Required packages to run this project is in requirements.txt

Step 1 - Run weightedknn.py(>>>python weightedknn.py)

Removes all the train data attributes which has more than 60% missing data.
Performs Weighted KNN Imputation on the resultant data and saves it in the file - knn_mode_imputed_data.csv
Uses the same resultant data and removes all the instances with missing data to create the train data without any imputation and is stored in file - non_imputed_full_data.csv
Removes all the missing data instances in test data and stores it in file - non_imputed_full_test_data.csv. 


Step 2 - Run feature_importance.py(>>>python2.7 feature_importance.py)

Creates four different dataset directories : 
	non_impute_feature_imp_results - no imputation , no smote
	non_impute_feature_imp_smote_results - no imputation , with smote
	weighted_knn_imputed_data - with imputation , no smote
	weighted_knn_smote_feature_imp - with imputation , with smote
	
	Under each directory, creates result file for the combination of feature selection method and classifier as - {feature_selection_method}-{classifier_method}.csv
	Under feature Importance directory in each dataset directory, creates file for ordering of best features for each feature selection methods as - {feature_selection_method}-importance.csv
	
Step 3 - Run summary_results.py(>>>python summary_results.py)

Requires summary_results.csv and top_features.csv for column headers. Will be replaced with data.

Creates a summary of results i.e. the best features selection method and classifiers for each dataset providing and saves it in the file - summary_results.csv
Lists the best feature combination for each feature selection for all four datasets and saves it in the file - top_features.csv

Step 4 - Run test_data.py(>>>python test_data.py)

Requires summary_results.csv with summary data for testing.
Requires knn_mode_imputed_data.csv, non_imputed_full_data.csv and  non_imputed_full_test_data.csv

Performs testing with test data and Melbourne health identified features on train and test data.
Summarises the results and saves it in file - test_data_summary_results.csv


Step 5 - Run plot_result.py(>>>python plot_result.py)

This script is present in each of the dataset directory. 
Running this script will analyse all the result files for that dataset and create plots under the "plots" directory.
Plots are saved as png images.
For each feature selection method for that dataset, Four different type of plots are stored in four different directories under "plots" directory
	"AUC_feature_comp" - plots of AUC vs # of features are stored as {feature_selection_method}-auc-no-of-feature.png
	"ACCURACY_feature_comp" - plots of Accuracy vs # of features are stored as {feature_selection_method}-accuracy-no-of-feature.png
	"AUC_PDF" - plots of probability distribution density vs AUC are stored as {feature_selection_method}-auc-prob-dist-func.png
	"ACCURACY_PDF" - plots of probability distribution density vs Accuracy are stored as {feature_selection_method}-accuracy-prob-dist-func.png
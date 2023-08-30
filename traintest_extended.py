# This file follows the seventh approach
# same as sixth but adding couples of points with respective weights

import pandas as pd
import xgboost as xgb
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import geopandas as gpd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

# Set path
os.chdir('/Users/naiacasina/Library/CloudStorage/OneDrive-UCB-O365/SEM2/B2P/Data/')

folder = "Rwanda"
country = "rwanda"
approach = 'seventh'
drop = "none"  # also: "primary", "secondary", "health", "semi_urban", "bridges", "population", "gdp"
combined = False

if combined:
    os.chdir('Combined/')
else:
    os.chdir(f'{folder}/')

# import train-test dataframe
data = pd.read_pickle(f'ML/{approach} approach/train_test_data_{approach}_combined_{combined}.pickle')
data = data.dropna()
inf_rows = data.isin([np.inf, -np.inf]).any(axis=1)
# filter out rows with inf values
data = data[~inf_rows].copy()
data['elev_perc_dif'] = data['elev_p75'] - data['elev_p25']

X = data.drop(columns=['label', 'geometry', 'nearest_distance_bridge', 'weight'])

# normalize the numerical features
columns_to_normalize = [
    'delta_time_df_primary_schools', 'max_time_df_primary_schools',
    'delta_time_df_secondary_schools','max_time_df_secondary_schools',
    'delta_time_df_health_centers', 'max_time_df_health_centers',
    'delta_time_df_semi_dense_urban','max_time_df_semi_dense_urban',
    'nearest_distance_footpath','nearest_distance_bridge',
    'pop_total','elevation_difference','elev_p25', 'elev_p50',
    'elev_p75','terrain_ruggedness','max_gdp', 'mean_gdp','elev_perc_dif'
]

# Create a MinMaxScaler instance
scaler = MinMaxScaler()
# Normalize the selected columns
data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])


# separate instances with 'weight' == 1 and 'weight' != 1
data_weighted = data[data['weight'] == 1]
data_non_weighted = data[data['weight'] != 1]

# split the instances with 'weight' == 1 into X_test_partial and y_test_partial
test_prop = 0.2
X_train_partial, X_test_partial, y_train_partial, y_test_partial = train_test_split(data_weighted.drop(['label', 'geometry', 'nearest_distance_bridge', 'weight'], axis=1), data_weighted['label'], test_size=test_prop, random_state=42)

# create X_train_all and y_train_all excluding the instances in X_test_partial
X_train_all = pd.concat([data_non_weighted.drop(['label', 'geometry', 'nearest_distance_bridge', 'weight'], axis=1), data_weighted.drop(['label', 'geometry', 'nearest_distance_bridge', 'weight'], axis=1)]).drop(X_test_partial.index)
y_train_all = pd.concat([data_non_weighted['label'], data_weighted['label']]).drop(X_test_partial.index)

# define the custom scoring metric
scoring = {'Precision': make_scorer(precision_score), 'Recall': make_scorer(recall_score),
           'f1': make_scorer(f1_score)}
scoring = make_scorer(f1_score)

# extract the 'weights' column for the training samples
train_weights = data.loc[X_train_all.index, 'weight']

# define classifiers with their respective hyperparameter spaces for grid search
classifiers = [
    ('XGBoost', xgb.XGBClassifier(),
     {'reg_alpha': [0, 0.1, 0.5], 'reg_lambda': [0, 0.1, 0.5], 'max_depth': [3, 5, 7], 'learning_rate': [0.1, 0.01],
      'subsample': [0.8, 1.0], 'gamma': [0, 0.2]}),
    ('Random Forest', RandomForestClassifier(),
     {'n_estimators': [100, 300], 'max_depth': [None, 5, 10], 'min_samples_split': [2, 5],
      'min_samples_leaf': [1, 3], 'max_features': ['sqrt', 'log2']})
]

performance = {'Classifier': [], 'Accuracy': [], 'Recall': [], 'Precision': []}
best_params = {}

# iterate over the classifiers, perform cross-validation, and evaluate each model
for name, clf, param_grid in classifiers:
    if name == 'XGBoost':
        # calculate class weights for XGBoost
        class_weights = len(y_train_all) / (2 * np.bincount(y_train_all))
        weight_ratio = class_weights[1] / class_weights[0]
        clf.set_params(scale_pos_weight=weight_ratio)
    elif name == 'RandomForest':
        # calculate class weights for RandomForest
        class_weights = len(y_train_all) / (2 * np.bincount(y_train_all))
        class_weight = {0: class_weights[0], 1: class_weights[1]}
        clf.set_params(class_weight=class_weight)

    # perform grid search with cross-validation
    grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5, scoring=scoring, refit='f1_score')
    grid_search.fit(X_train_all, y_train_all, sample_weight=train_weights)  # Use train_weights here

    # use the best estimator from the grid search
    best_clf = grid_search.best_estimator_
    best_params[name] = grid_search.best_params_

    # perform cross-validation on the best estimator
    cv_scores = cross_val_score(best_clf, X_train_all, y_train_all, cv=5)

    # make predictions on the test set
    y_pred = best_clf.predict(X_test_partial)

    accuracy = accuracy_score(y_test_partial, y_pred)
    recall = recall_score(y_test_partial, y_pred)
    precision = precision_score(y_test_partial, y_pred)

    performance['Classifier'].append(name)
    performance['Accuracy'].append(accuracy)
    performance['Recall'].append(recall)
    performance['Precision'].append(precision)

    print(f"Classifier: {name}")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Cross-Validation Scores: {cv_scores}")
    print(f"Test Accuracy: {accuracy}")
    print(f"Test Recall: {recall}")
    print(f"Test Precision: {precision}")
    print("-------------------------------------")

# convert performance dictionary into a df
performance_df = pd.DataFrame(performance)
# save the performance_df dictionary to a pickle file
with open(f'Saved data/performance_{approach}_test_size_{test_prop}_dropped_{drop}.pkl', 'wb') as f:
    pickle.dump(performance_df, f)

with open(f'Saved data/best_params_{approach}_test_size_{test_prop}.pkl', 'wb') as f:
    pickle.dump(best_params, f)

# ----- Plotting feature importances -----
with open(f'Saved data/performance_{approach}_test_size_{test_prop}_dropped_{drop}.pkl', 'rb') as f:
    performance_df = pickle.load(f)
with open(f'Saved data/best_params_{approach}_test_size_{test_prop}.pkl', 'rb') as f:
    best_params = pickle.load(f)
# set a color palette
colors = sns.color_palette('pastel')

# create a figure and subplots
fig, axs = plt.subplots(len(classifiers), 1, figsize=(8, (len(classifiers)) * 4))

# iterate over the classifiers
for i, (name, clf, param_grid) in enumerate(classifiers):
    # exclude the SVM, KNN, and Neural Network classifiers
    if name in ['SVM', 'KNN', 'Neural Network']:
        continue

    # get the feature importances for the current classifier
    try:
        # get the best parameters for the current classifier
        best_params_clf = best_params[name]

        # create an instance of the classifier with the best parameters
        clf_best = clf.set_params(**best_params_clf)
        clf_best.fit(X_train_all, y_train_all)
        feature_importances = clf_best.feature_importances_
        feature_names = X.columns

        # plot feature importances in the corresponding subplot
        axs[i].bar(feature_names, feature_importances, color=colors)
        axs[i].set_xlabel('Features' if i == len(classifiers) - 4 else '')
        axs[i].set_ylabel('Importance')
        axs[i].set_title(f'Feature Importances - {name}')
        axs[i].tick_params(axis='x', rotation=90)
        axs[i].grid(axis='y', linestyle='--', alpha=0.5)

        # omit x labels in the top subplot
        if i != len(classifiers) - 1:
            axs[i].set_xticklabels([])
    except AttributeError:
        # skip classifiers without feature importances
        continue

# adjust spacing between subplots
plt.tight_layout()
# save the figure
plt.savefig(f'ML/Performance plots/feature_importances_{approach}_test_size_{test_prop}.png')
# show the plot
plt.show()


#---------- Spatial spread of the error ------------
with open(f'Saved data/best_params_{approach}_test_size_{test_prop}.pkl', 'rb') as f:
    best_params = pickle.load(f)
# create a dictionary to store the results
results = {}
# create an additional dictionary to store instances with error > 0.5
high_error_instances = {}

# iterate over the classifiers and their best parameters
for i, (name, clf, param_grid) in enumerate(classifiers):
    # exclude the SVM, KNN, and Neural Network classifiers
    if name in ['SVM', 'KNN', 'Neural Network']:
        continue

    best_params_clf = best_params[name]
    # create an instance of the classifier with the best parameters
    clf_best = clf.set_params(**best_params_clf)
    clf_best.fit(X_train_all, y_train_all)

    # predict class probabilities on the entire dataset
    y_pred_proba = clf_best.predict_proba(X_train_all)[:, 1]

    # calculate the absolute error for each instance
    absolute_error = np.abs(y_train_all - y_pred_proba)
    if name=='XGBoost':
        # add the 'absolute_error' column to the data DataFrame
        data['absolute_error'] = absolute_error

    # create a DataFrame with the geometry and absolute error
    error_df = pd.DataFrame({'geometry': data['geometry'], 'absolute_error': absolute_error})
    # create a GeoDataFrame from the error DataFrame
    error_gdf = gpd.GeoDataFrame(error_df, geometry='geometry')
    error_gdf.to_file(f"Shapefiles/{name}_errors.shp")
    # filter instances with error > 0.5
    high_error_instances[name] = data[absolute_error > 0.5]
    # store the error GeoDataFrame for the current classifier
    results[name] = error_gdf

with open(f'Saved data/spatial_errors_{approach}.pkl', 'wb') as f:
    pickle.dump(results, f)
# store the high-error instances dictionary
with open(f'Saved data/high_error_instances_{approach}.pkl', 'wb') as f:
    pickle.dump(high_error_instances, f)







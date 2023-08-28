import pandas as pd
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
import seaborn as sns
import os
import geopandas as gpd

folder = "Rwanda"
country = "rwanda"
approach = 'sixth'
drop = "none"  # also: "primary", "secondary", "health", "semi_urban", "bridges", "population", "gdp"

os.chdir(f"/Users/naiacasina/Library/CloudStorage/OneDrive-UCB-O365/SEM2/B2P/Data/{folder}/")

# import train-test dataframe
data = pd.read_pickle(f'ML/{approach} approach/train_test_data_{approach}.pickle')
data = data.dropna()
inf_rows = data.isin([np.inf, -np.inf]).any(axis=1)
# filter out rows with inf values
data = data[~inf_rows].copy()

data['elev_perc_dif'] = data['elev_p75'] - data['elev_p25']

# assuming your DataFrame is called 'data'
X = data.drop(['label', 'geometry', 'nearest_distance_bridge'], axis=1)
y = data['label']  # Target variable

# normalize the numerical features
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

test_prop = 0.2
# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=test_prop, random_state=42)
# define the custom scoring metric
scoring = {'Precision': make_scorer(precision_score), 'Recall': make_scorer(recall_score),
           'f1': make_scorer(f1_score)}
scoring = make_scorer(f1_score)


# define classifiers with their respective hyperparameter spaces for grid search
classifiers = [
    ('XGBoost', xgb.XGBClassifier(),
     {'reg_alpha': [0, 0.1, 0.5], 'reg_lambda': [0, 0.1, 0.5], 'max_depth': [3, 5, 7], 'learning_rate': [0.1, 0.01],
      'subsample': [0.8, 1.0], 'gamma': [0, 0.1, 0.2]}),
    ('Random Forest', RandomForestClassifier(),
     {'n_estimators': [100, 300], 'max_depth': [None, 5, 10], 'min_samples_split': [2, 5],
      'min_samples_leaf': [1, 3], 'max_features': ['sqrt', 'log2']}),
    ('SVM', SVC(), {'C': [1, 10, 100], 'kernel': ['linear', 'rbf']}),
    ('KNN', KNeighborsClassifier(), {'n_neighbors': [3, 5, 7]}),
    ('Neural Network', MLPClassifier(), {'alpha': [0.0001, 0.001, 0.01],
                                         'hidden_layer_sizes': [(64,), (128,), (64, 64)],
                                         'solver': ['adam', 'sgd'],
                                         'batch_size': [32, 64]})
]


performance = {'Classifier': [], 'Accuracy': [], 'Recall': [], 'Precision': []}
best_params = {}

# iterate over the classifiers, perform cross-validation, and evaluate each model
for name, clf, param_grid in classifiers:
    if name == 'XGBoost':
        # calculate class weights for XGBoost
        class_weights = len(y_train) / (2 * np.bincount(y_train))
        weight_ratio = class_weights[1] / class_weights[0]
        clf.set_params(scale_pos_weight=weight_ratio)

    # perform grid search with cross-validation
    grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5, scoring=scoring, refit='f1_score')
    grid_search.fit(X_train, y_train)

    # use the best estimator from the grid search
    best_clf = grid_search.best_estimator_
    best_params[name] = grid_search.best_params_

    # perform cross-validation on the best estimator
    cv_scores = cross_val_score(best_clf, X_train, y_train, cv=5)

    # make predictions on the test set
    y_pred = best_clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

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

# ----- Partial dependence plots -----
# train above with the following:
classifiers = [
    ('XGBoost', xgb.XGBClassifier(),
     {'reg_alpha': [0, 0.1, 0.5], 'reg_lambda': [0, 0.1, 0.5], 'max_depth': [1], 'learning_rate': [0.1, 0.01],
      'subsample': [0.8, 1.0], 'gamma': [0, 0.1, 0.2]})
]

# assuming X_train and y_train are the training data
feature_names = X.columns.to_list()

best_params_clf = best_params['XGBoost']
clf = xgb.XGBClassifier()
# create an instance of the classifier with the best parameters
clf_best = clf.set_params(**best_params_clf)
# fit the XGBoost classifier with max_depth=1
clf_best.fit(X_train, y_train)

features = [0, 2, 4, 6, 8, 9, 10, 16]

# compute partial dependence plots
display = PartialDependenceDisplay.from_estimator(clf_best, X_train, features, feature_names=feature_names)
# customize plot appearance
fig, ax = plt.subplots(figsize=(14, 6))  # Set the figure size
display.plot(ax=ax, n_cols=4)  # Set the number of columns for subplots

# add a title
ax.set_title('Partial dependence plots', fontsize=16)
# modify x-axis label size
ax.set_xlabel('Feature Values', fontsize=8)
# remove y-axis labels
ax.set_ylabel('')
# make x-axis ticks smaller
ax.tick_params(axis='x', labelsize=6)
# add more separation between subplots
plt.subplots_adjust(wspace=0.3, hspace=0.75)

# save the plot with good resolution
plt.savefig(f'ML/Partial plots/partial_dependence_plots_{approach}.png', dpi=300)
# show the plot
plt.show()


# ----- Plotting feature importances -----
with open(f'Saved data/performance_{approach}_test_size_{test_prop}_dropped.pkl', 'rb') as f:
    performance_df = pickle.load(f)
with open(f'Saved data/best_params_{approach}_test_size_{test_prop}.pkl', 'rb') as f:
    best_params = pickle.load(f)
# Set a color palette
colors = sns.color_palette('pastel')

# create a figure and subplots
fig, axs = plt.subplots(len(classifiers) - 3, 1, figsize=(8, (len(classifiers) - 3) * 4))

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
        clf_best.fit(X_train, y_train)
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
        if i != len(classifiers) - 4:
            axs[i].set_xticklabels([])
    except AttributeError:
        continue

# adjust spacing between subplots
plt.tight_layout()
# save the figure
plt.savefig(f'ML/Performance plots/feature_importances_{approach}_test_size_{test_prop}.png')
# show the plot
plt.show()

# --------- Leaving one out for the training and testing ---------
from sklearn.metrics import accuracy_score, precision_score, recall_score

with open(f'Saved data/best_params_{approach}_test_size_{test_prop}.pkl', 'rb') as f:
    best_params = pickle.load(f)

# train XGBoost classifier with all features
xgb_clf = xgb.XGBClassifier(**best_params['XGBoost'])
xgb_clf.fit(X_train, y_train)

# make predictions on the test set using the trained model
y_pred_all_features = xgb_clf.predict(X_test)
accuracy_all_features = accuracy_score(y_test, y_pred_all_features)
precision_all_features = precision_score(y_test, y_pred_all_features)
recall_all_features = recall_score(y_test, y_pred_all_features)

# initialize the XGBoost classifier with the best parameters
best_xgb = xgb.XGBClassifier(**best_params['XGBoost'])

# create empty dictionaries to store the metrics for each feature
accuracy_scores = {}
precision_scores = {}
recall_scores = {}

# iterate over each feature and evaluate the model after dropping it
for feature in X.columns:
    # drop the current feature from the dataset
    X_dropped = X.drop(feature, axis=1)

    # normalize the numerical features
    X_dropped_normalized = scaler.fit_transform(X_dropped)

    # split the data into training and testing sets
    X_train_dropped, X_test_dropped, y_train, y_test = train_test_split(
        X_dropped_normalized, y, test_size=test_prop, random_state=42
    )

    # fit the XGBoost classifier on the training data
    best_xgb.fit(X_train_dropped, y_train)

    # make predictions on the test set
    y_pred = best_xgb.predict(X_test_dropped)

    # calculate and store the metrics
    accuracy_scores[feature] = accuracy_score(y_test, y_pred)
    precision_scores[feature] = precision_score(y_test, y_pred)
    recall_scores[feature] = recall_score(y_test, y_pred)

# convert the metric dictionaries into dataframes
accuracy_df = pd.DataFrame.from_dict(accuracy_scores, orient='index', columns=['Accuracy'])
precision_df = pd.DataFrame.from_dict(precision_scores, orient='index', columns=['Precision'])
recall_df = pd.DataFrame.from_dict(recall_scores, orient='index', columns=['Recall'])

# plot the metrics for each dropped feature
fig, axes = plt.subplots(3, 1, figsize=(10, 12))

# set the minimum y-range to 0.5
y_min = 0.7
y_max = 0.9
accuracy_df.plot(kind='bar', ax=axes[0], legend=False)
axes[0].set_ylabel('Accuracy')
axes[0].set_title('XGBoost Performance after Dropping Features')
axes[0].set_ylim(y_min, y_max)  # Set the y-range for accuracy plot

precision_df.plot(kind='bar', ax=axes[1], legend=False)
axes[1].set_ylabel('Precision')
axes[1].set_ylim(y_min, y_max)  # Set the y-range for precision plot

recall_df.plot(kind='bar', ax=axes[2], legend=False)
axes[2].set_ylabel('Recall')
axes[2].set_ylim(0.5, 0.7)  # Set the y-range for recall plot

for ax, metric in zip(axes, ['Accuracy', 'Precision', 'Recall']):
    all_features_value = accuracy_all_features if metric == 'Accuracy' else \
                         precision_all_features if metric == 'Precision' else \
                         recall_all_features
    ax.axhline(y=all_features_value, color='green', linestyle='dashed', label='All Features')


# hide x-axis labels on the first two plots
axes[0].xaxis.set_ticklabels([])
axes[1].xaxis.set_ticklabels([])

# set x-axis labels on the third plot
axes[2].set_xlabel('Dropped Feature')

plt.xlabel('Dropped Feature')

plt.tight_layout()
plt.savefig(f'ML/Performance plots/feature_drop_{approach}.png', dpi=300, bbox_inches='tight')
plt.show()

# sort the dataframes in descending order by index (y-axis features)
accuracy_df.sort_index(ascending=False, inplace=True)
precision_df.sort_index(ascending=False, inplace=True)
recall_df.sort_index(ascending=False, inplace=True)

# plot the metrics for each dropped feature
fig, axes = plt.subplots(1, 3, figsize=(12, 6))

# set the minimum y-range to 0.5
y_min = 0.7
y_max = 0.9

# plot the accuracy values
accuracy_df.plot(kind='barh', ax=axes[0], legend=False)
axes[0].set_xlabel('Accuracy')
axes[0].set_title('XGBoost Performance after Dropping Features')
axes[0].set_xlim(y_min, y_max)  # Set the x-range for accuracy plot

# plot the precision values
precision_df.plot(kind='barh', ax=axes[1], legend=False)
axes[1].set_xlabel('Precision')
axes[1].set_xlim(y_min, y_max)  # Set the x-range for precision plot

# plot the recall values
recall_df.plot(kind='barh', ax=axes[2], legend=False)
axes[2].set_xlabel('Recall')
axes[2].set_xlim(0.5, 0.7)  # Set the x-range for recall plot

# add vertical lines for the all-feature values
for ax, metric in zip(axes, ['Accuracy', 'Precision', 'Recall']):
    all_features_value = accuracy_all_features if metric == 'Accuracy' else \
                         precision_all_features if metric == 'Precision' else \
                         recall_all_features
    ax.axvline(x=all_features_value, color='green', linestyle='dashed', label='All Features')

# hide y-axis labels on the first two plots
axes[1].set_yticklabels([])
axes[2].set_yticklabels([])

# set y-axis labels on the third plot
axes[0].set_ylabel('Dropped Feature')

# adjust spacing
plt.tight_layout()
# save the plot with good quality
plt.savefig(f'ML/Performance plots/feature_drop_{approach}.png', dpi=300, bbox_inches='tight')
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
    clf_best.fit(X_normalized, y)

    # predict class probabilities on the entire dataset
    y_pred_proba = clf_best.predict_proba(X_normalized)[:, 1]

    # calculate the absolute error for each instance
    absolute_error = np.abs(y - y_pred_proba)
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



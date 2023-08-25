# Overview

This repository offers a comprehensive toolkit for constructing and evaluating models designed to identify potential remote bridge sites. The models are built using a range of features, including walk time variations between different geographical points across multiple countries, elevation metrics, population and GDP data, distances to existing bridges, and footpath availability.

The project employs a systematic approach to model selection and parameter tuning. A grid search is performed across various machine learning algorithms, such as XGBoost, Random Forest, Support Vector Machines (SVM), k-Nearest Neighbors (KNN), and Neural Networks (NN). This process automates the selection of optimal hyperparameters, enhancing the overall model performance.

Performance metrics, including accuracy, recall, and precision, are evaluated for each model. Additionally, a set of powerful tools is provided for thorough analysis. These tools include partial dependence plots, feature importances, leave-one-out cross-validation, spatial error distribution visualization, and performance plots.

With this repository, users can leverage a robust set of tools and models to address the challenging task of identifying potential remote bridge sites. By integrating various features and employing diverse machine learning techniques, the project aims to enhance the accuracy and effectiveness of such identification processes.

# Script contents

This section outlines the functionality of the different scripts included in the repository:

1. `main.py`: Extracts features for positive labels (known bridge sites) and negative labels (randomly selected waterway points) across the area covered by bridge sites.
2. `train_test.py`: Performs a grid search across various models with the train/test dataset of features plus labels obtained from `main.py`, selects optimal parameters, and calculates performance metrics.
3. `main_extended.py` and `train_test_extended.py`: Extended versions of the above scripts that augment the training set by adding points to both sides of the label-coordinates, with downweighting based on distance.
4. `deployment.py`: Takes trained classifiers and deploys the model to all segments of waterways in a specified region.
5. `performance_analysis.py`: Provides comprehensive analysis tools, including partial dependence plots, feature importances, leave-one-out cross-validation, and spatial error distribution visualization.
6. `performance_plots.py`: Reads performance metrics, generates, and saves comparison plots.
7. `linestrings_to_points.py`: Densifies waterway linestrings and extracts points at chosen regular intervals.
8. `eliminate_bridges_outof_ww.py`: Filters out bridges that are not in or close to waterways, enhancing the quality of the training data.
9. `get_osm_data.py`: Retrieves bridge sites, schools, and health centres data from OpenStreetMap (OSM).
10. `raster_processing.py`: Sets a consistent extent and resolution for different rasters within a country, and merges elevation rasters.

Each script contributes a specific functionality to the workflow, encompassing data extraction, model training, deployment, performance analysis, and more. The provided scripts collectively enable the identification of remote bridge sites through a robust and systematic approach.

# Running

To get started, follow these steps to set up and run the scripts:

1. **Download Repository and Data Folders**: Clone or download this repository along with the necessary data folders from the provided shared OneDrive link.

2. **Configure Data Path**:
   - Navigate to the downloaded repository folder, call it *name_of_folder*.
   - Open each script in a text editor of your choice.\
   - Change the path to point to the location of the downloaded data folder (*name_of_folder*) at the beginning of each script.

3. **Run the Scripts**:
   - With the path variable updated, you can now execute any script by running it from the repository folder.
   - For example, you can run `python main.py` to execute the feature extraction script.

4. **Demo Script**:
   - The `demo.py` script showcases the process using the extended training set and provides performance plot visualizations.
   - Run the demo script using `python demo.py`.
   - The script uses `train_test_extended.py` and data from `main_extended.py` for Rwanda.
   - It also generates a shapefile of Point probability predictions in the folder `Rwanda/Shapefiles`.

By following these steps, you'll be able to utilize the provided scripts to identify remote bridge sites effectively. The demo script offers a quick way to observe the entire workflow with extended training data and visualized results.


# Main output

Upon running the scripts, the repository generates several key output files that contribute to analysis and deployment:

1. **Performance Analysis Outputs**:
   - Performance analysis output files are located at `*country_name*/ML/Performance plots`.
   - These files encompass visualization tools such as partial dependence plots, feature importances, leave-one-out cross-validation, and spatial error distribution visualizations.
   
2. **Deployment Outputs**:
   - Deployment-related output files are found in the folder `*country_name*/Shapefiles`.
   - The following outputs are available:
     - Point Probability Predictions: Shapefile containing predictions for points along waterways.
     - Filtered Bridge Locations: Shapefile of bridge locations filtered to those in or close to waterways.
     - Sampled Waterway Points: Sampled points from waterways for model input.
    
*country_name* can be changed by *combined*, which takes into account the merger of training points from several full-covered regions. So far, the whole country of Rwanda and Kabarole, Kasese, Ibanda, & Bundibugyo districts of Uganda.

# This file takes trained classifiers and deploys the model in all segments of waterways in a country

import pandas as pd
import xgboost as xgb
import geopandas as gpd
import pickle
import warnings
import os
import numpy as np
from shapely.geometry import MultiPolygon, Polygon, Point, LineString
from shapely import wkb
import rasterio
from rasterio.mask import mask
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from pyproj import CRS


approach = "seventh"
country = "rwanda"
folder = "Rwanda"

os.chdir(f'/Users/naiacasina/Library/CloudStorage/OneDrive-UCB-O365/SEM2/B2P/Data/{folder}')

# ww points
ww = gpd.read_file(f'{country}_waterways_osm_shape/{country}_waterways_osm_shape.shp')
ww.crs = "EPSG:4326"
# subregion polygons
subregions = pd.read_parquet(f'{country}_subregions.parquet')
# read the "subregions" dataframe
subregions_gdf = gpd.GeoDataFrame(subregions)
# decode the binary representation and create geometries
subregions_gdf['geometry'] = subregions_gdf['geometry'].apply(lambda x: wkb.loads(x, hex=True))
subregions_gdf = subregions_gdf.set_geometry('geometry')
subregions_gdf.crs = "EPSG:4326"

# travel times: merged_dfs dictionary
with open('Saved data/merged_dfs.pickle', 'rb') as file:
    merged_dfs = pickle.load(file)

projected_crs = 'EPSG:4326' 
subregions_gdf = subregions_gdf.to_crs(projected_crs)

# footpaths
footpaths = pd.read_parquet('footpaths.parquet')
footpaths_gdf = gpd.GeoDataFrame(footpaths)
# decode the binary representation and create geometries
footpaths_gdf['geometry'] = footpaths_gdf['geometry'].apply(lambda x: wkb.loads(x, hex=True))
footpaths_gdf = footpaths_gdf.set_geometry('geometry')

with rasterio.open("population.tif") as src_pop:
    population_raster = src_pop.read(1)
    population_raster = np.where(population_raster == -99999, 0, population_raster)  # Set no data value to NaN

with rasterio.open("Rasters/elevation.tif") as src:
    elevation_raster = src.read(1)
    elevation_transform = src.transform

radius = 16
buffer_radius = 0.0045  # Initial buffer radius (start with a smaller value)
# define the projected CRS that you want to use for distance calculations
projected_crs = 'EPSG:4326'

# ---------------- NON BRIDGE POINTS ----------------
# set the random seed for reproducibility
random.seed(42)
# ww points
ww = gpd.read_file(f'Shapefiles/ww_to_points.shp')
ww.crs = "EPSG:4326"
bridges_gdf = gpd.read_file(f'Shapefiles/filtered_bridge_locations.shp')
bridges_gdf.crs = "EPSG:4326"
admin_units_gdf = gpd.read_file(f'rwanda_admin_boundaries/rwanda_admin_1.shp')
admin_units_gdf.crs = "EPSG:4326"
bridges_with_units = gpd.sjoin(bridges_gdf, admin_units_gdf, predicate='within')
bridge_counts = bridges_with_units.groupby('NAME_1').size()
# calculate the ratio of bridges in each administrative unit
total_bridges = bridge_counts.sum()
bridge_ratios = bridge_counts / total_bridges
# perform a spatial join to associate waterways points with administrative units
ww_with_units = gpd.sjoin(ww, admin_units_gdf, predicate='within')

# create an empty list to store the sampled Point geometries
sampled_points = []
# define the minimum distance in degrees (0.0045 degrees)
min_distance_degrees = 0.00277
# function to check if the Point is at least 'min_distance_degrees' away from other points
def is_min_distance_away(point, other_points):
    for other_point in other_points:
        if point.distance(other_point) < min_distance_degrees:
            return False
    return True

# keep sampling until we have 10000 points
while len(sampled_points) < 10000:
    # take a random administrative unit with probability proportional to the bridge ratio
    random_unit = random.choices(bridge_ratios.index, weights=bridge_ratios.values)[0]
    # filter the waterways dataset to only include points in the selected administrative unit
    ww_filtered = ww_with_units[ww_with_units['NAME_1'] == random_unit]
    # take a random row from the filtered waterways dataset
    random_row = ww_filtered.sample(n=1)
    point = random_row['geometry'].iloc[0]
    print(len(sampled_points))

    # check if the Point is at least 'min_distance_degrees' away from other points
    if is_min_distance_away(point, sampled_points):
        sampled_points.append(point)

# create a new GeoDataFrame with the sampled Point geometries
ww_gdf = gpd.GeoDataFrame(geometry=sampled_points, crs=ww.crs)
projected_crs = CRS.from_epsg(4326)
# reproject the GeoDataFrames to the projected CRS
ww_gdf.crs = "EPSG:4326"
ww_gdf = ww_gdf.to_crs(projected_crs)
subregions_gdf = subregions_gdf.to_crs(projected_crs)
subregions_gdf.crs = "EPSG:4326"
ww_gdf.to_file(filename="Shapefiles/deployment_points.shp", driver="ESRI Shapefile")


# -------------- Load gdf --------------
ww_gdf = gpd.read_file(filename="Shapefiles/deployment_points.shp")
ww_gdf.crs = "EPSG:4326"
subregions_gdf.crs = "EPSG:4326"

for key, df in merged_dfs.items():
    ww_gdf[f'delta_time_{key}'] = None
    ww_gdf[f'max_time_{key}'] = None
    print(key)

    for index, row in ww_gdf.iterrows():
        bridge_geometry = row.geometry
        bridge_buffer = bridge_geometry.buffer(0.001)  # Adjust the buffer distance as needed

        try:
            # find the first polygon that intersects with the bridge
            first_intersection = subregions_gdf[subregions_gdf.intersects(bridge_buffer)].iloc[0]
            # find the nearest polygon to the bridge that is not the first intersection
            # perform the distance calculations
            nearest_polygon_index = subregions_gdf[
                ~subregions_gdf.intersects(bridge_buffer)].distance(bridge_geometry).idxmin()
            nearest_polygon = subregions_gdf.iloc[nearest_polygon_index]

            # save the subregion indices
            first_subregion_index = first_intersection['subregion_index']
            nearest_subregion_index = nearest_polygon['subregion_index']
        except IndexError:
            # handle the case when no intersection is found
            # drop the row from ww_gdf and continue with the next
            ww_gdf = ww_gdf.drop(index)
            continue

        filtered_schools = df[df['subregion'] == first_subregion_index]

        try:
            # find the nearest point
            distances = filtered_schools['geometry'].distance(bridge_geometry)

            if distances.empty:
                continue

            # get the index of the nearest point
            nearest_index = distances.idxmin()

            # get the nearest point and its travel time
            nearest_point_1 = filtered_schools.loc[nearest_index, 'geometry']
            nearest_travel_time_1 = filtered_schools.loc[nearest_index, 'travel_time']
        except ValueError:
            nearest_travel_time_1 = np.inf
        # filter df by subregion
        filtered_schools = df[df['subregion'] == nearest_subregion_index]

        try:
            # find the nearest point
            distances = filtered_schools['geometry'].distance(bridge_geometry)
            if distances.empty:
                continue
            # get the index of the nearest point
            nearest_index = distances.idxmin()
            # get the nearest point and its travel time
            nearest_point_2 = filtered_schools.loc[nearest_index, 'geometry']
            nearest_travel_time_2 = filtered_schools.loc[nearest_index, 'travel_time']
        except ValueError:
            nearest_travel_time_2 = np.inf

        # calculate time differences and maximum times
        ww_gdf.at[index, f'delta_time_{key}'] = abs(nearest_travel_time_1 - nearest_travel_time_2)
        ww_gdf.at[index, f'max_time_{key}'] = max(nearest_travel_time_1, nearest_travel_time_2)

for index, row in ww_gdf.iterrows():
    bridge_geometry = row['geometry']
    print(index)

    # DISTANCE TO NEAREST BRIDGE AND FOOTPATH
    # compute distance to nearest footpath
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    distances_footpath = footpaths_gdf['geometry'].distance(bridge_geometry)
    warnings.resetwarnings()

    nearest_distance_footpath = distances_footpath.min()
    # compute distance to nearest bridge (excluding the current bridge)
    distances_bridge = ww_gdf[ww_gdf.index != index]['geometry'].distance(bridge_geometry)
    nearest_distance_bridge = distances_bridge.min()

    # update the respective columns in bridges_gdf
    ww_gdf.at[index, 'nearest_distance_footpath'] = nearest_distance_footpath
    ww_gdf.at[index, 'nearest_distance_bridge'] = nearest_distance_bridge

    # POPULATION COUNT
    # create a buffer around the bridge point
    buffer_geom = bridge_geometry.buffer(buffer_radius)
    # find the intersecting polygons
    intersecting_polygons = subregions_gdf[subregions_gdf.intersects(buffer_geom)].copy()

    # check if there are more than two intersecting polygons
    if len(intersecting_polygons) > 2:
        # compute intersection areas for each polygon
        intersecting_polygons['intersection_area'] = intersecting_polygons.intersection(buffer_geom).area
        # sort the polygons by intersection area in descending order
        intersecting_polygons = intersecting_polygons.sort_values('intersection_area', ascending=False)
        # select the first two polygons with the greatest intersection area
        intersecting_polygons = intersecting_polygons.head(2)

    # calculate population count within the intersecting polygons
    population_count = []
    gdp_count = []
    if len(intersecting_polygons) <= 1:
        ww_gdf.drop(index, inplace=True)
        continue

    try:
        # iterate over the intersecting polygons
        for poly in intersecting_polygons.geometry:
            # compute the intersection between the polygon and the buffer
            intersection_geom = poly.intersection(buffer_geom)

            if isinstance(intersection_geom, MultiPolygon):
                if isinstance(intersection_geom, MultiPolygon):
                    # find the polygon with the maximum area
                    largest_area = max(p.area for p in intersection_geom.geoms)
                    largest_polygon = next(p for p in intersection_geom.geoms if p.area == largest_area)
                    intersection_geom = largest_polygon

            # check if there is a valid intersection
            if not intersection_geom.is_empty:
                # convert the intersection geometry to a list of polygons
                intersection_polygons = [Polygon(intersection_geom)]

                # open the raster dataset
                with rasterio.open("population.tif") as dataset:
                    # mask the population raster using the intersection polygons
                    masked_data, _ = mask(dataset, intersection_polygons, crop=True)
                    masked_data = np.where(masked_data == -99999, 0, masked_data)
                    # calculate the population sum within the masked area
                    population_sum = masked_data.sum()

                    # add the population sum to the total count
                    population_count.append(population_sum)

                    with rasterio.open("Wealth/GDP2005_1km.tif") as dataset:
                        # mask the population raster using the intersection polygons
                        masked_data, _ = mask(dataset, intersection_polygons, crop=True)
                        # calculate the population sum within the masked area
                        gdp_mean = masked_data.mean()
                        # add the population sum to the total count
                        gdp_count.append(gdp_mean)
    except ValueError:
        print(ValueError)
        continue

    sorted_counts = sorted(population_count, reverse=True)

    max_count = sorted_counts[0]
    second_max_count = sorted_counts[1] if len(sorted_counts) > 1 else 0
    total_count = max_count + second_max_count

    # get indices of the pop counts for the GDP
    max_index = population_count.index(max_count)
    second_max_index = population_count.index(second_max_count)
    max_gdp = gdp_count[max_index]
    mean_gdp = (gdp_count[max_index] + gdp_count[second_max_index]) / 2

    # update the corresponding columns in bridges_df
    ww_gdf.at[index, "pop_total"] = total_count
    ww_gdf.at[index, "pop_ratio_max"] = max_count / total_count
    ww_gdf.at[index, "max_gdp"] = max_gdp
    ww_gdf.at[index, "mean_gdp"] = mean_gdp

    # ELEVATION
    # convert bridge coordinates to pixel indices
    x_bridge, y_bridge = bridge_geometry.x, bridge_geometry.y
    # compute the inverse of the elevation_transform
    elevation_transform_inv = ~elevation_transform
    # transform the bridge coordinates to pixel coordinates
    pixel_coords = elevation_transform_inv * (x_bridge, y_bridge)
    x, y = int(pixel_coords[0]), int(pixel_coords[1])

    elevation_bridge = elevation_raster[int(y), int(x)]
    y_min = max(0, int(y - radius))
    x_min = max(0, int(x - radius))
    y_max = min(20000, int(y + radius + 1))
    x_max = min(20000, int(x + radius + 1))

    surrounding_elevations = elevation_raster[y_min:y_max, x_min:x_max]
    elevation_difference = elevation_bridge - np.mean(surrounding_elevations)
    # elevation percentiles
    elevation_values = surrounding_elevations.flatten()
    elevation_percentiles = np.percentile(elevation_values, [25, 50, 75])
    # compute slope
    dx, dy = np.gradient(surrounding_elevations)
    slope = np.sqrt(dx ** 2 + dy ** 2)
    # compute terrain ruggedness
    terrain_ruggedness = np.std(slope)

    ww_gdf.at[index, "elevation_difference"] = elevation_difference
    ww_gdf.at[index, "elev_p25"] = elevation_percentiles[0]
    ww_gdf.at[index, "elev_p50"] = elevation_percentiles[1]
    ww_gdf.at[index, "elev_p75"] = elevation_percentiles[2]
    ww_gdf.at[index, "terrain_ruggedness"] = terrain_ruggedness

# save the negative labels to a pickle file
with open('Saved data/deployment_instances.pickle', 'wb') as file:
    pickle.dump(ww_gdf, file)

# save deployment geometries to shapefile
ww_gdf[["geometry"]].to_file(f"Shapefiles/{country}_deployment_points.shp", driver="ESRI Shapefile")

# ----------------------------------------------------
# load the best classifier and train in all data
test_prop = 0.2
os.chdir(f"/Users/naiacasina/Library/CloudStorage/OneDrive-UCB-O365/SEM2/B2P/Data/{folder}/")
with open(f'Saved data/best_params_{approach}_test_size_{test_prop}.pkl', 'rb') as f:
    best_params = pickle.load(f)

with open(f'Saved data/deployment_instances.pickle', 'rb') as f:
    ww_gdf = pickle.load(f)

# import train-test dataframe
data = pd.read_pickle(f'ML/{approach} approach/train_test_data_{approach}.pickle')
data = data.dropna()
inf_rows = data.isin([np.inf, -np.inf]).any(axis=1)
# filter out rows with inf values
data = data[~inf_rows].copy()
data['elev_perc_dif'] = data['elev_p75'] - data['elev_p25']
X = data.drop(['label', 'geometry', 'nearest_distance_bridge', 'weight'], axis=1)
y = data['label']  # Target variable
# normalize the numerical features
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# deployment points
ww_gdf = ww_gdf.dropna()
inf_rows = ww_gdf.isin([np.inf, -np.inf]).any(axis=1)
# filter out rows with inf values
ww_gdf = ww_gdf[~inf_rows].copy()
ww_gdf['elev_perc_dif'] = ww_gdf['elev_p75'] - ww_gdf['elev_p25']
# ww_gdf = ww_gdf.drop(['pop_ratio_max'], axis=1)
# add exact bridge points
data_label_1 = data[data['label'] == 1].drop(['label'], axis=1)
ww_gdf = pd.concat([ww_gdf, data_label_1], ignore_index=True)
X_dep = ww_gdf.drop(['geometry', 'nearest_distance_bridge', 'weight'], axis=1)
# normalize deployment instances
X_dep_normalized = scaler.fit_transform(X_dep)

classifiers = [
    ('XGBoost', xgb.XGBClassifier(),
     {'reg_alpha': [0, 0.1, 0.5], 'reg_lambda': [0, 0.1, 0.5], 'max_depth': [3, 5, 7], 'learning_rate': [0.1, 0.01],
      'subsample': [0.8, 1.0], 'gamma': [0, 0.1, 0.2]}),
    ('Random Forest', RandomForestClassifier(),
     {'n_estimators': [100, 300], 'max_depth': [None, 5, 10], 'min_samples_split': [2, 5],
      'min_samples_leaf': [1, 3], 'max_features': ['sqrt', 'log2']})
]

# train in all the data
for i, (name, clf, param_grid) in enumerate(classifiers):

    # get the best parameters for the current classifier
    best_params_clf = best_params[name]

    # create an instance of the classifier with the best parameters
    clf_best = clf.set_params(**best_params_clf)
    clf_best.fit(X_normalized, y)

    # predict class probabilities on the entire dataset
    y_pred_proba = clf_best.predict_proba(X_dep_normalized)[:, 1]

    predict_df = pd.DataFrame({'geometry': ww_gdf['geometry'], 'predicted_proba': y_pred_proba})
    # create a GeoDataFrame from the error DataFrame
    predict_df = gpd.GeoDataFrame(predict_df, geometry='geometry')
    predict_df.to_file(f"Shapefiles/{name}_predictions.shp")



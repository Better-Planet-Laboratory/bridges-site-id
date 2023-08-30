# This file incorporates the outcomes from Matthew's model: travel times
# As an additional feature to the model, I take the Delta_t for each infrastructure
# for each positive or negative label (e.g., for each bridge or non bridge site) as
# well as the greatest travel time out of the two.
# I also consider the additional features as in the previous models:
# elevation metrics (mean, percentiles, terrain ruggedness), population (total, ratio).
# Additional feature incorporated: GDP
# Additional changes to the code: population and GDP are taken from the polygon intersections with the buffer
# that make up the greatest area (if there are more than 2 polygons within the buffer)

import pandas as pd
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

# Set path
os.chdir('path_to_folder')

approach = "sixth"
country = "rwanda"
folder = "Rwanda"
approach = "sixth"
os.chdir(f'/{folder}')

# subregion polygons
subregions = pd.read_parquet(f'{country}_subregions.parquet')
subregions_gdf = gpd.GeoDataFrame(subregions)
# decode the binary representation and create geometries
subregions_gdf['geometry'] = subregions_gdf['geometry'].apply(lambda x: wkb.loads(x, hex=True))
subregions_gdf = subregions_gdf.set_geometry('geometry')
subregions_gdf.crs = "EPSG:4326"

# bridge points
bridges = pd.read_parquet(f'b2p_{country}_bridges.parquet')
# convert bridge coordinates to pygeos Point objects
bridges_gdf = gpd.read_file(f'Shapefiles/filtered_bridge_locations.shp')
bridges_gdf.crs = "EPSG:4326"

# ww points
ww = gpd.read_file(f'Shapefiles/ww_to_points.shp')
ww.crs = "EPSG:4326"

if country=="uganda":
    # perform a spatial join to filter waterways and bridges
    ww_filtered = gpd.sjoin(ww, subregions_gdf, how='inner', predicate='intersects')
    bridges_filtered = gpd.sjoin(bridges_gdf, subregions_gdf, how='inner', predicate='intersects')
    ww_filtered[["geometry"]].to_file("Shapefiles/subregion_filtered_ww.shp", driver="ESRI Shapefile")

# travel times: merged_dfs dictionary
with open('Saved data/merged_dfs.pickle', 'rb') as file:
    merged_dfs = pickle.load(file)

projected_crs = 'EPSG:4326'
subregions_gdf = subregions_gdf.to_crs(projected_crs)
bridges_gdf = bridges_gdf.to_crs(projected_crs)

for key, df in merged_dfs.items():
    # create columns for the time differences and maximum times
    bridges_gdf[f'delta_time_{key}'] = None
    bridges_gdf[f'max_time_{key}'] = None
    print(key)

    for index, row in bridges_gdf.iterrows():
        bridge_geometry = row.geometry
        bridge_buffer = bridge_geometry.buffer(0.001)  # Adjust the buffer distance as needed

        try:
            # find the first polygon that intersects with the bridge
            first_intersection = subregions_gdf[subregions_gdf.intersects(bridge_buffer)].iloc[0]
        except IndexError:
            # if there is no intersection, skip to the next bridge
            continue

        # find the nearest polygon to the bridge that is not the first intersection
        nearest_polygon = subregions_gdf[~subregions_gdf.intersects(bridge_buffer)].distance(bridge_geometry).idxmin()
        nearest_polygon = subregions_gdf.loc[nearest_polygon]

        # save the subregion indices
        first_subregion_index = first_intersection['subregion_index']
        nearest_subregion_index = nearest_polygon['subregion_index']

        # filter df by subregion
        filtered_schools = df[df['subregion'] == first_subregion_index]

        try:
            # find the nearest point
            distances = filtered_schools['geometry'].distance(bridge_geometry)

            if distances.empty:
                raise ValueError('No schools found in filtered subset')

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
                raise ValueError('No schools found in filtered subset')

            # get the index of the nearest point
            nearest_index = distances.idxmin()

            # get the nearest point and its travel time
            nearest_point_2 = filtered_schools.loc[nearest_index, 'geometry']
            nearest_travel_time_2 = filtered_schools.loc[nearest_index, 'travel_time']
        except ValueError:
            nearest_travel_time_2 = np.inf

        # calculate time differences and maximum times
        bridges_gdf.at[index, f'delta_time_{key}'] = abs(nearest_travel_time_1 - nearest_travel_time_2)
        bridges_gdf.at[index, f'max_time_{key}'] = max(nearest_travel_time_1, nearest_travel_time_2)

# footpaths
footpaths = pd.read_parquet(f'{country}_footpaths_osm.parquet')
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
buffer_radius = 0.0045  # initial buffer radius (start with a smaller value)
# define the projected CRS that you want to use for distance calculations
projected_crs = 'EPSG:4326'  

# reproject the GeoDataFrame and bridge_geometry to the projected CRS
bridges_gdf = bridges_gdf.to_crs(projected_crs)

for index, row in bridges_gdf.iterrows():
    bridge_geometry = row['geometry']
    print(index)

    # DISTANCE TO NEAREST BRIDGE AND FOOTPATH
    # compute distance to nearest footpath
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    distances_footpath = footpaths_gdf['geometry'].distance(bridge_geometry)
    warnings.resetwarnings()

    nearest_distance_footpath = distances_footpath.min()

    # compute distance to nearest bridge (excluding the current bridge)
    distances_bridge = bridges_gdf[bridges_gdf.index != index]['geometry'].distance(bridge_geometry)
    nearest_distance_bridge = distances_bridge.min()

    # update the respective columns in bridges_gdf
    bridges_gdf.at[index, 'nearest_distance_footpath'] = nearest_distance_footpath
    bridges_gdf.at[index, 'nearest_distance_bridge'] = nearest_distance_bridge


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

    try:
        # calculate population count within the intersecting polygons
        population_count = []
        gdp_count = []

        if len(intersecting_polygons) <= 1:
            bridges_gdf.drop(index, inplace=True)
            continue
        try:

            # iterate over the intersecting polygons
            for poly in intersecting_polygons.geometry:
                # compute the intersection between the polygon and the buffer
                intersection_geom = poly.intersection(buffer_geom)

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
        mean_gdp = (gdp_count[max_index] + gdp_count[second_max_index])/2

        # gpdate the corresponding columns in bridges_df
        bridges_gdf.at[index, "pop_total"] = total_count
        bridges_gdf.at[index, "pop_ratio_max"] = max_count/total_count
        bridges_gdf.at[index, "max_gdp"] = max_gdp
        bridges_gdf.at[index, "mean_gdp"] = mean_gdp


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

        bridges_gdf.at[index, "elevation_difference"] = elevation_difference
        bridges_gdf.at[index, "elev_p25"] = elevation_percentiles[0]
        bridges_gdf.at[index, "elev_p50"] = elevation_percentiles[1]
        bridges_gdf.at[index, "elev_p75"] = elevation_percentiles[2]
        bridges_gdf.at[index, "terrain_ruggedness"] = terrain_ruggedness
    except IndexError as e:
        print("Caught INdexError:", e)
        continue

# save the merged_dfs dictionary to a pickle file
with open('Saved data/positive_labels.pickle', 'wb') as file:
    pickle.dump(bridges_gdf, file)

with open('Saved data/positive_labels.pickle', 'rb') as f:
    bridges_gdf = pickle.load(f)

# ---------------- NON BRIDGE POINTS ----------------
# set the random seed for reproducibility
random.seed(42)

sampled_points = []
# define the minimum distance in degrees (0.0045 degrees)
min_distance_degrees = 0.00277
# function to check if the Point is at least 'min_distance_degrees' away from other points
def is_min_distance_away(point, other_points):
    for other_point in other_points:
        if point.distance(other_point) < min_distance_degrees:
            return False
    return True

# keep sampling without replacement until we have 3000 points
while len(sampled_points) < 3000:
    # take a random row from 'ww' without replacement
    random_row = ww.sample(n=1)
    point = random_row['geometry'].iloc[0]
    print(len(sampled_points))
    # check if the Point is at least 'min_distance_degrees' away from other points
    if is_min_distance_away(point, sampled_points):
        sampled_points.append(point)

# create a new GeoDataFrame with the sampled Point geometries
ww_gdf = gpd.GeoDataFrame(geometry=sampled_points, crs=ww.crs)

from pyproj import CRS
projected_crs = CRS.from_epsg(4326)
ww_gdf.crs = "EPSG:4326"
ww_gdf = ww_gdf.to_crs(projected_crs)
subregions_gdf = subregions_gdf.to_crs(projected_crs)
subregions_gdf.crs = "EPSG:4326"

for key, df in merged_dfs.items():
    # create columns for the time differences and maximum times
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
    ww_gdf.at[index, "pop_ratio_max"] = max_count/total_count
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
with open('Saved data/negative_labels.pickle', 'wb') as file:
    pickle.dump(ww_gdf, file)


# -------- MERGER ---------
with open('Saved data/negative_labels.pickle', 'rb') as f:
    ww_gdf = pickle.load(f)
with open('Saved data/positive_labels.pickle', 'rb') as f:
    bridges_gdf = pickle.load(f)

bridges_gdf.drop('pop_ratio_max', axis=1, inplace=True)
ww_gdf.drop('pop_ratio_max', axis=1, inplace=True)
ww_gdf = ww_gdf.dropna()

# add 'label' column with values 1 to bridges_gdf
bridges_gdf['label'] = 1
# get the desired number of entries for ww_gdf (twice the size of bridges_gdf)
desired_size = 3 * len(bridges_gdf)
# check if ww_gdf has more entries than the desired size
if len(ww_gdf) > desired_size:
    # randomly select the desired number of rows from ww_gdf
    ww_gdf_sampled = ww_gdf.sample(n=desired_size, random_state=42)
else:
    # if ww_gdf has fewer entries, keep all of its rows
    ww_gdf_sampled = ww_gdf
ww_gdf_sampled['label'] = 0

# merge the two GeoDataFrames based on the specified columns
merged_gdf = pd.concat([bridges_gdf, ww_gdf_sampled], ignore_index=True)
# list of columns to merge
merge_columns = [
    'geometry',
    'delta_time_df_primary_schools',
    'max_time_df_primary_schools',
    'delta_time_df_secondary_schools',
    'max_time_df_secondary_schools',
    'delta_time_df_health_centers',
    'max_time_df_health_centers',
    'delta_time_df_semi_dense_urban',
    'max_time_df_semi_dense_urban',
    'nearest_distance_footpath',
    'nearest_distance_bridge',
    'pop_total',
    'elevation_difference',
    'elev_p25',
    'elev_p50',
    'elev_p75',
    'terrain_ruggedness', 'max_gdp', 'mean_gdp',
    'label'
]

# perform the merge on the specified columns
merged_gdf = merged_gdf[merge_columns]

# save the traintest dataset to a pickle file
merged_gdf.to_pickle(f'ML/{approach} approach/train_test_data_{approach}.pickle')


# This file incorporates the outcomes from Matthew's model: travel times
# Same as sixth approach, but takes additional couples of points to both
# sides of the positive and negative label-coordinates, downweighting them
# The downweight and the number of couples of points can be chosen manually

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

approach = "seventh"
country = "rwanda"
folder = "Rwanda"
approach = "seventh"
combined = True

# check if combined regions
if combined:
    os.chdir(f'/Users/naiacasina/Library/CloudStorage/OneDrive-UCB-O365/SEM2/B2P/Data/')
    os.chdir('Combined')
else:
    os.chdir(f'/Users/naiacasina/Library/CloudStorage/OneDrive-UCB-O365/SEM2/B2P/Data/')
    os.chdir(f'{folder}')

if combined:
    # list of folder-country pairs
    folder_country_pairs = [
        ('Rwanda', 'rwanda'),
        ('Uganda', 'uganda'),
    ]

    # initialize lists to store GeoDataFrames
    subregions_gdfs = []
    bridges_gdfs = []
    ww_gdfs = []

    # initialize dictionaries to store merged_dfs from different folders
    all_merged_dfs = []

    # loop through folder-country pairs
    for folder, country in folder_country_pairs:
        folder_path = f'/Users/naiacasina/Library/CloudStorage/OneDrive-UCB-O365/SEM2/B2P/Data/{folder}'
        os.chdir(folder_path)

        # read subregions
        subregions = pd.read_parquet(f'{country}_subregions.parquet')
        subregions_gdf = gpd.GeoDataFrame(subregions)
        subregions_gdf['geometry'] = subregions_gdf['geometry'].apply(lambda x: wkb.loads(x, hex=True))
        subregions_gdf = subregions_gdf.set_geometry('geometry')
        subregions_gdf.crs = "EPSG:4326"
        subregions_gdfs.append(subregions_gdf)

        # read and set CRS for bridges
        bridges_gdf = gpd.read_file(f'Shapefiles/filtered_bridge_locations.shp')
        bridges_gdf.crs = "EPSG:4326"
        bridges_gdfs.append(bridges_gdf)

        # read and set CRS for waterways
        ww = gpd.read_file(f'Shapefiles/ww_to_points.shp')
        ww.crs = "EPSG:4326"
        ww_gdfs.append(ww)

        # load the merged_dfs dictionary from the pickle file
        with open('Saved data/merged_dfs.pickle', 'rb') as file:
            merged_dfs = pickle.load(file)

        # add the loaded merged_dfs to the all_merged_dfs dictionary
        all_merged_dfs.append(merged_dfs)

    subregions = gpd.GeoDataFrame(pd.concat(subregions_gdfs, ignore_index=True))
    bridges_gdf = gpd.GeoDataFrame(pd.concat(bridges_gdfs, ignore_index=True))
    ww = gpd.GeoDataFrame(pd.concat(ww_gdfs, ignore_index=True))

else:
    # subregion polygons
    subregions = pd.read_parquet(f'{country}_subregions.parquet')
    # read the "subregions" dataframe
    subregions_gdf = gpd.GeoDataFrame(subregions)
    # decode the binary representation and create geometries
    subregions_gdf['geometry'] = subregions_gdf['geometry'].apply(lambda x: wkb.loads(x, hex=True))
    subregions_gdf = subregions_gdf.set_geometry('geometry')
    subregions_gdf.crs = "EPSG:4326"

    # filtered bridges
    bridges_gdf = gpd.read_file(f'Shapefiles/filtered_bridge_locations.shp')
    bridges_gdf.crs = "EPSG:4326"

    # point waterways
    ww = gpd.read_file(f'Shapefiles/ww_to_points.shp')
    ww.crs = "EPSG:4326"

    # Travel times: merged_dfs dictionary
    with open('Saved data/merged_dfs.pickle', 'rb') as file:
        merged_dfs = pickle.load(file)


projected_crs = 'EPSG:4326'
subregions_gdf = subregions_gdf.to_crs(projected_crs)
bridges_gdf = bridges_gdf.to_crs(projected_crs)

# initialize an empty dictionary to store the concatenated DataFrames for each key
concatenated_dfs = {}

# iterate over the keys in the first dictionary to initialize the concatenated_dfs dictionary
for key in all_merged_dfs[0]:
    concatenated_dfs[key] = []

# concatenate DataFrames for each key across dictionaries
for data_dict in all_merged_dfs:
    for key, df in data_dict.items():
        concatenated_dfs[key].append(df)

# merge concatenated DataFrames for each key
merged_dfs = {}
for key, df_list in concatenated_dfs.items():
    merged_dfs[key] = pd.concat(df_list, ignore_index=True)

# -------- Extending bridges_gdf with ww points -----------
# define the number of closest pairs to consider
num_closest_pairs = 6
# calculate weights based on the number of closest pairs
weights = [0.8 - (0.2 * (i // 2)) for i in range(num_closest_pairs)]
# initialize a list to store new rows for bridges_gdf
new_rows = []

# loop through each bridge point
for bridge_idx, bridge_row in bridges_gdf.iterrows():
    bridge_point = bridge_row['geometry']
    # calculate distances to all points in ww
    distances = ww['geometry'].distance(bridge_point)
    # sort distances and get the indices of closest points
    closest_indices = distances.argsort()[:num_closest_pairs]
    # create dictionaries for each row
    bridge_dict = {
        'geometry': bridge_point,
        'weight': 1.0
    }
    new_rows.extend([bridge_dict])

    # loop through each pair of closest points
    for i in range(0, num_closest_pairs, 2):
        closest_point1 = ww.loc[closest_indices[i], 'geometry']
        closest_point2 = ww.loc[closest_indices[i + 1], 'geometry']

        # calculate weight for the pair
        weight = weights[i]

        closest_point1_dict = {
            'geometry': closest_point1,
            'weight': weight
        }
        closest_point2_dict = {
            'geometry': closest_point2,
            'weight': weight
        }

        # append dictionaries to the list
        new_rows.extend([closest_point1_dict, closest_point2_dict])

# create a new GeoDataFrame from the list of dictionaries
result_gdf = gpd.GeoDataFrame(new_rows)
# set the geometry column to Point objects
result_gdf['geometry'] = result_gdf['geometry'].apply(Point)
# set the CRS for the GeoDataFrame
result_gdf.crs = bridges_gdf.crs

for key, df in merged_dfs.items():
    result_gdf[f'delta_time_{key}'] = None
    result_gdf[f'max_time_{key}'] = None
    print(key)

    for index, row in result_gdf.iterrows():
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
        result_gdf.at[index, f'delta_time_{key}'] = abs(nearest_travel_time_1 - nearest_travel_time_2)
        result_gdf.at[index, f'max_time_{key}'] = max(nearest_travel_time_1, nearest_travel_time_2)

footpaths_gdfs = []
population_rasters = []
elevation_rasters = []

if combined:
    # loop through folder-country pairs
    for folder, country in folder_country_pairs:
        folder_path = f'/Users/naiacasina/Library/CloudStorage/OneDrive-UCB-O365/SEM2/B2P/Data/{folder}'
        os.chdir(folder_path)

        # read footpaths
        footpaths = pd.read_parquet(f'{country}_footpaths_osm.parquet')
        footpaths_gdf = gpd.GeoDataFrame(footpaths)
        footpaths_gdf['geometry'] = footpaths_gdf['geometry'].apply(lambda x: wkb.loads(x, hex=True))
        footpaths_gdf = footpaths_gdf.set_geometry('geometry')
        footpaths_gdfs.append(footpaths_gdf)

        # read population raster
        with rasterio.open(f"population.tif") as src_pop:
            population_raster = src_pop.read(1)
            population_raster = np.where(population_raster == -99999, 0, population_raster)  # Set no data value to NaN
            population_rasters.append(population_raster)

        # read elevation raster
        with rasterio.open(f"Rasters/elevation.tif") as src:
            elevation_raster = src.read(1)
            elevation_transform = src.transform
            elevation_rasters.append(elevation_raster)
    footpaths = gpd.GeoDataFrame(pd.concat(footpaths_gdfs, ignore_index=True))
    population = np.concatenate(population_rasters, axis=0)  # Concatenate along axis 0
    elevation = np.concatenate(elevation_rasters, axis=0)  # Concatenate along axis 0

else:
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
buffer_radius = 0.0045  # Initial buffer radius (start with a smaller value)
projected_crs = 'EPSG:4326'
result_gdf = result_gdf.to_crs(projected_crs)

for index, row in result_gdf.iterrows():
    bridge_geometry = row['geometry']
    print(index)

    # DISTANCE TO NEAREST BRIDGE AND FOOTPATH
    # compute distance to nearest footpath
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    distances_footpath = footpaths_gdf['geometry'].distance(bridge_geometry)
    warnings.resetwarnings()

    nearest_distance_footpath = distances_footpath.min()

    # compute distance to nearest bridge (excluding the current bridge)
    distances_bridge = result_gdf[result_gdf.index != index]['geometry'].distance(bridge_geometry)
    nearest_distance_bridge = distances_bridge.min()

    # update the respective columns in bridges_gdf
    result_gdf.at[index, 'nearest_distance_footpath'] = nearest_distance_footpath
    result_gdf.at[index, 'nearest_distance_bridge'] = nearest_distance_bridge

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
            result_gdf.drop(index, inplace=True)
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
        mean_gdp = (gdp_count[max_index] + gdp_count[second_max_index]) / 2

        # update the corresponding columns in bridges_df
        result_gdf.at[index, "pop_total"] = total_count
        result_gdf.at[index, "pop_ratio_max"] = max_count / total_count
        result_gdf.at[index, "max_gdp"] = max_gdp
        result_gdf.at[index, "mean_gdp"] = mean_gdp

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

        result_gdf.at[index, "elevation_difference"] = elevation_difference
        result_gdf.at[index, "elev_p25"] = elevation_percentiles[0]
        result_gdf.at[index, "elev_p50"] = elevation_percentiles[1]
        result_gdf.at[index, "elev_p75"] = elevation_percentiles[2]
        result_gdf.at[index, "terrain_ruggedness"] = terrain_ruggedness
    except IndexError as e:
        print("Caught INdexError:", e)
        continue

# save the merged_dfs dictionary to a pickle file
with open(f'Saved data/positive_labels_{combined}.pickle', 'wb') as file:
    pickle.dump(result_gdf, file)

with open(f'Saved data/positive_labels_{combined}.pickle', 'rb') as f:
    result_gdf = pickle.load(f)

# ---------------- NON BRIDGE POINTS ----------------
# set the random seed for reproducibility
random.seed(42)

# read admin-level shp
all_admin_units = gpd.GeoDataFrame()
if combined:
    for folder, country in folder_country_pairs:
        admin_units_gdf = gpd.read_file(f'/Users/naiacasina/Library/CloudStorage/OneDrive-UCB-O365/SEM2/B2P/Data/{folder}/{country}_admin_boundaries/{country}_admin_2.shp')
        all_admin_units = pd.concat([all_admin_units, admin_units_gdf], ignore_index=True)
else:
    all_admin_units = gpd.read_file(f'/Users/naiacasina/Library/CloudStorage/OneDrive-UCB-O365/SEM2/B2P/Data/{folder}/{country}_admin_boundaries/{country}_admin_2.shp')

bridges_with_units = gpd.sjoin(bridges_gdf, all_admin_units, predicate='within')
bridge_counts = bridges_with_units.groupby('NAME_2').size()
# calculate the ratio of bridges in each admin-2 unit
total_bridges = bridge_counts.sum()
bridge_ratios = bridge_counts / total_bridges
# pre-process: create a dictionary to store filtered datasets by admin-2 unit
filtered_datasets = {}
for unit in bridge_ratios.index:
    filtered_datasets[unit] = ww[ww.within(all_admin_units[all_admin_units['NAME_2'] == unit].geometry.iloc[0])]

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


# keep sampling without replacement until we have 3000 points
while len(sampled_points) < 3000:
    # take a random admin-2 unit with probability proportional to the bridge ratio
    random_unit = random.choices(bridge_ratios.index, weights=bridge_ratios.values)[0]
    # take a random row from the filtered waterways dataset for the selected admin-2 unit
    ww_filtered = filtered_datasets[random_unit]
    random_row = ww_filtered.sample(n=1)
    point = random_row['geometry'].iloc[0]
    print(len(sampled_points))
    # check if the Point is at least 'min_distance_degrees' away from other points
    if is_min_distance_away(point, sampled_points):
        sampled_points.append(point)

# create a new GeoDataFrame with the sampled Point geometries
ww_gdf = gpd.GeoDataFrame(geometry=sampled_points, crs=ww.crs)

ww_gdf.to_file(filename=f"/Users/naiacasina/Library/CloudStorage/OneDrive-UCB-O365/SEM2/B2P/Data/Combined/Shapefiles/ww_sampled.shp", driver="ESRI Shapefile")

# ---------- Extend the ww dataset by adding ------------
# ---------- surrounding couples of points --------------
# initialize a list to store new rows for ww
ww_rows = []

# loop through each bridge point
for ww_idx, ww_row in ww_gdf.iterrows():
    ww_point = ww_row['geometry']
    print(ww_idx)

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    # calculate distances to all points in ww
    distances = ww['geometry'].distance(ww_point)
    warnings.resetwarnings()

    # sort distances and get the indices of closest points
    # add + 1 because the closest point is the ww point itself
    closest_indices = distances.argsort()[:(num_closest_pairs+1)]

    # create dictionaries for each row
    ww_dict = {
        'geometry': ww_point,
        'weight': 1.0
    }
    ww_rows.extend([ww_dict])

    # loop through each pair of closest points
    for i in range(1, num_closest_pairs+1, 2):
        closest_point1 = ww.loc[closest_indices[i], 'geometry']
        closest_point2 = ww.loc[closest_indices[i + 1], 'geometry']

        # calculate weight for the pair
        weight = weights[i]
        closest_point1_dict = {
            'geometry': closest_point1,
            'weight': weight
        }
        closest_point2_dict = {
            'geometry': closest_point2,
            'weight': weight
        }
        # append dictionaries to the list
        ww_rows.extend([closest_point1_dict, closest_point2_dict])

# create a new GeoDataFrame from the list of dictionaries
ww_result_gdf = gpd.GeoDataFrame(ww_rows)
# set the geometry column to Point objects
ww_result_gdf['geometry'] = ww_result_gdf['geometry'].apply(Point)
# set the CRS for the GeoDataFrame
ww_result_gdf.crs = ww_gdf.crs

from pyproj import CRS
projected_crs = CRS.from_epsg(4326)
# reproject the GeoDataFrames to the projected CRS
ww_result_gdf.crs = "EPSG:4326"
ww_result_gdf = ww_result_gdf.to_crs(projected_crs)
subregions_gdf = subregions_gdf.to_crs(projected_crs)
subregions_gdf.crs = "EPSG:4326"

for key, df in merged_dfs.items():
    ww_result_gdf[f'delta_time_{key}'] = None
    ww_result_gdf[f'max_time_{key}'] = None
    print(key)

    for index, row in ww_result_gdf.iterrows():
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
            ww_result_gdf = ww_result_gdf.drop(index)
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
        ww_result_gdf.at[index, f'delta_time_{key}'] = abs(nearest_travel_time_1 - nearest_travel_time_2)
        ww_result_gdf.at[index, f'max_time_{key}'] = max(nearest_travel_time_1, nearest_travel_time_2)

for index, row in ww_result_gdf.iterrows():
    bridge_geometry = row['geometry']
    print(index)

    # DISTANCE TO NEAREST BRIDGE AND FOOTPATH
    # compute distance to nearest footpath
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    distances_footpath = footpaths_gdf['geometry'].distance(bridge_geometry)
    warnings.resetwarnings()

    nearest_distance_footpath = distances_footpath.min()

    # compute distance to nearest bridge (excluding the current bridge)
    distances_bridge = ww_result_gdf[ww_result_gdf.index != index]['geometry'].distance(bridge_geometry)
    nearest_distance_bridge = distances_bridge.min()

    # update the respective columns in bridges_gdf
    ww_result_gdf.at[index, 'nearest_distance_footpath'] = nearest_distance_footpath
    ww_result_gdf.at[index, 'nearest_distance_bridge'] = nearest_distance_bridge

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
        ww_result_gdf.drop(index, inplace=True)
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
    ww_result_gdf.at[index, "pop_total"] = total_count
    ww_result_gdf.at[index, "pop_ratio_max"] = max_count / total_count
    ww_result_gdf.at[index, "max_gdp"] = max_gdp
    ww_result_gdf.at[index, "mean_gdp"] = mean_gdp

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
    y_max = int(y + radius + 1)
    x_max = int(x + radius + 1)

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

    ww_result_gdf.at[index, "elevation_difference"] = elevation_difference
    ww_result_gdf.at[index, "elev_p25"] = elevation_percentiles[0]
    ww_result_gdf.at[index, "elev_p50"] = elevation_percentiles[1]
    ww_result_gdf.at[index, "elev_p75"] = elevation_percentiles[2]
    ww_result_gdf.at[index, "terrain_ruggedness"] = terrain_ruggedness

# save the negative labels to a pickle file
with open(f'Saved data/negative_labels_{combined}.pickle', 'wb') as file:
    pickle.dump(ww_result_gdf, file)

# -------- MERGED -------
with open(f'Saved data/negative_labels_{combined}.pickle', 'rb') as f:
    ww_result_gdf = pickle.load(f)
with open(f'Saved data/positive_labels_{combined}.pickle', 'rb') as f:
    result_gdf = pickle.load(f)

result_gdf.drop('pop_ratio_max', axis=1, inplace=True)
ww_result_gdf.drop('pop_ratio_max', axis=1, inplace=True)
ww_result_gdf = ww_result_gdf.dropna()

# add 'label' column with values 1 to bridges_gdf
result_gdf['label'] = 1
# get the desired number of entries for ww_gdf (twice the size of bridges_gdf)
desired_size = 3 * len(result_gdf)
# check if ww_gdf has more entries than the desired size
if len(ww_result_gdf) > desired_size:
    # randomly select the desired number of rows from ww_gdf
    ww_gdf_sampled = ww_result_gdf.sample(n=desired_size, random_state=42)
else:
    # If ww_gdf has fewer entries, keep all of its rows
    ww_gdf_sampled = ww_result_gdf
ww_gdf_sampled = ww_result_gdf
ww_gdf_sampled['label'] = 0

# merge the two GeoDataFrames based on the specified columns
merged_gdf = pd.concat([result_gdf, ww_gdf_sampled], ignore_index=True)
# list of columns to merge
merge_columns = [
    'geometry',
    'weight',
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
    'label'  # Including 'label' column in merge columns
]

# perform the merge on the specified columns
merged_gdf = merged_gdf[merge_columns]

# save the traintest dataset to a pickle file
merged_gdf.to_pickle(f'ML/{approach} approach/train_test_data_{approach}_combined_{combined}.pickle')


import geopandas as gpd
import os
import pandas as pd
from shapely import wkb

approach = "seventh"
country = "uganda"
folder = "Uganda"
approach = "seventh"

os.chdir(f'/Users/naiacasina/Library/CloudStorage/OneDrive-UCB-O365/SEM2/B2P/Data/')
os.chdir(f'{folder}')

gdf = gpd.read_parquet(f'Waterways/{country}_waterways_osm.parquet')

if country == "uganda":
    # subregion polygons
    subregions = pd.read_parquet(f'{country}_subregions.parquet')
    # read the "subregions" dataframe
    subregions_gdf = gpd.GeoDataFrame(subregions)
    # decode the binary representation and create geometries
    subregions_gdf['geometry'] = subregions_gdf['geometry'].apply(lambda x: wkb.loads(x, hex=True))
    subregions_gdf = subregions_gdf.set_geometry('geometry')
    subregions_gdf.crs = "EPSG:4326"
    gdf.crs = "EPSG:4326"
    ww_filtered = gpd.sjoin(gdf, subregions_gdf, how='inner', predicate='intersects')

gdf = ww_filtered.drop_duplicates(subset=['name', 'geometry'])
# function to densify the LineString and extract points at regular intervals
def densify_line(line, distance):
    points = []
    for i in range(int(line.length / distance) + 1):
        point = line.interpolate(i * distance)
        points.append(point)
    return points

# define the distance (30 meters or 0.00027 degrees)
distance_meters = 30
distance_degrees = 0.00027
# create an empty list to store Point geometries
point_geometries = []

# loop through each LineString in the GeoDataFrame
for index, row in gdf.iterrows():
    line = row['geometry']
    points = densify_line(line, distance_degrees)
    point_geometries.extend(points)
    print(index)

# create a new GeoDataFrame with the Point geometries
gdf_points = gpd.GeoDataFrame(geometry=point_geometries, crs=gdf.crs)
gdf_points.to_file(filename="Shapefiles/ww_to_points.shp",driver="ESRI Shapefile")



# -------
# Get bridge sites, schools, health centers from OSM using API
# -------
import os
import requests
import geopandas as gpd
from shapely.geometry import Point

# Set path
os.chdir('/Users/naiacasina/Documents/SEM2/B2P/Data/')

folder = "Ethiopia"
country = "ethiopia"
os.chdir(f'{folder}/')

# ------------------------ BRIDGES ------------------------
# define the Overpass API query
overpass_url = "https://lz4.overpass-api.de/api/interpreter"  # URL of the Overpass API server

# query to get all bridges
query = """
    [out:json];
    area["name"="Uganda"]->.a;
    (
        way["bridge"="yes"](area.a);
        relation["bridge"="yes"](area.a);
    );
    out center;
"""

# send the request to the Overpass API
response = requests.get(overpass_url, params={"data": query})

# parse the response as JSON
data = response.json()

# extract bridge locations from the response
bridge_locations = []
for element in data["elements"]:
    if element["type"] == "node":
        bridge_locations.append((element["lon"], element["lat"]))
    elif element["type"] == "way" or element["type"] == "relation":
        bridge_locations.append((element["center"]["lon"], element["center"]["lat"]))

# print the bridge locations
for location in bridge_locations:
    print(location)

# create a list of Shapely Point geometries from the bridge locations
bridge_points = [Point(lon, lat) for lon, lat in bridge_locations]

# create a GeoDataFrame with the bridge points
bridge_gdf = gpd.GeoDataFrame(geometry=bridge_points)

# define the output shapefile path
output_shapefile = f'Shapefiles/bridge_locations.shp'

# save the GeoDataFrame as a shapefile
bridge_gdf.to_file(output_shapefile)


# ----- PARQUET TO SHAPEFILE ------
# read the parquet file into a GeoDataFrame
gdf = gpd.read_parquet(f"/{country}_bridges_osm.parquet")
gdf['timestamp'] = gdf['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
# Save the GeoDataFrame as a shapefile
gdf.to_file(output_shapefile)



query = """
    [out:json];
    area["name"="Uganda"]->.a;
    (
        way["bridge"="yes"](area.a);
        relation["bridge"="yes"](area.a);
    );
    out center;
"""


# ------------------------ SCHOOLS ------------------------
# define the Overpass API query to retrieve primary and secondary schools in the specified country
# define the country code
country_code = "ET"
# UG: Uganda
# ET: Ethiopia

# define the Overpass API query
query = f"""
[out:json][timeout:25];
area["ISO3166-1"="{country_code}"]->.searchArea;
(
  node["amenity"="school"](area.searchArea);
  way["amenity"="school"](area.searchArea);
  relation["amenity"="school"](area.searchArea);
);
out center;
"""

# send the request to the Overpass API
response = requests.get(f"https://overpass-api.de/api/interpreter?data={query}")

# convert the JSON response to a GeoDataFrame
data = response.json()

school_locations = []
school_types = []
for element in data["elements"]:
    if element["type"] == "node" and "lat" in element and "lon" in element and "tags" in element:
        lat = element["lat"]
        lon = element["lon"]
        school_locations.append((lon, lat))
        school_name = element["tags"].get("name", "")
        if "primary" in school_name.lower():
            school_type = "Primary School"
        elif "secondary" in school_name.lower():
            school_type = "Secondary School"
        else:
            school_type = ""
        school_types.append(school_type)

# create a GeoDataFrame from the school locations and types
schools_gdf = gpd.GeoDataFrame(
    {"school_type": school_types},
    geometry=gpd.points_from_xy([lon for lon, lat in school_locations], [lat for lon, lat in school_locations])
)

# set the coordinate reference system (CRS) if known
schools_gdf.crs = "EPSG:4326"  # WGS84 coordinate system

output_shapefile = f'{country}_education_facilities/education_facilities.shp'

# save the GeoDataFrame to a shapefile
schools_gdf.to_file(output_shapefile)



# ------------------ HEALTH CENTERS -------------
import requests
import geopandas as gpd

# define the country code for Uganda
country_code = "UG"

# define the Overpass API query for health centers
query = f"""
[out:json][timeout:25];
area["ISO3166-1"="{country_code}"]->.searchArea;
(
  node["amenity"="clinic"](area.searchArea);
  node["amenity"="hospital"](area.searchArea);
  node["amenity"="health_post"](area.searchArea);
  node["amenity"="doctors"](area.searchArea);
  node["amenity"="pharmacy"](area.searchArea);
  node["amenity"="dentist"](area.searchArea);
  node["amenity"="laboratory"](area.searchArea);
  node["amenity"="blood_donation"](area.searchArea);
  node["amenity"="optician"](area.searchArea);
  node["amenity"="veterinary"](area.searchArea);
  node["amenity"="alternative"](area.searchArea);
);
out center;
"""

# send the request to the Overpass API
response = requests.get(f"https://overpass-api.de/api/interpreter?data={query}")

# convert the JSON response to a GeoDataFrame
data = response.json()

health_center_locations = []
health_center_types = []
for element in data["elements"]:
    if element["type"] == "node" and "lat" in element and "lon" in element and "tags" in element:
        lat = element["lat"]
        lon = element["lon"]
        health_center_locations.append((lon, lat))
        health_center_name = element["tags"].get("name", "")
        health_center_type = element["tags"].get("amenity", "")
        health_center_types.append(health_center_type)

# create a GeoDataFrame from the health center locations and types
health_centers_gdf = gpd.GeoDataFrame(
    {"health_center_type": health_center_types},
    geometry=gpd.points_from_xy([lon for lon, lat in health_center_locations], [lat for lon, lat in health_center_locations])
)

# set the coordinate reference system (CRS) if known
health_centers_gdf.crs = "EPSG:4326"  # WGS84 coordinate system

output_shapefile = f'{country}_health_facilities/health_facilities.shp'

# save the GeoDataFrame to a shapefile
health_centers_gdf.to_file(output_shapefile)

# --------- RELIGIOUS FACILITIES -----------
# define the country code for Uganda
country_code = "UG"

# define the Overpass API query for religious facilities
query = f"""
[out:json][timeout:25];
area["ISO3166-1"="{country_code}"]->.searchArea;
(
  node["amenity"="place_of_worship"](area.searchArea);
);
out center;
"""

# send the request to the Overpass API
response = requests.get(f"https://overpass-api.de/api/interpreter?data={query}")

# convert the JSON response to a GeoDataFrame
data = response.json()

religious_locations = []
religious_types = []
for element in data["elements"]:
    if element["type"] == "node" and "lat" in element and "lon" in element and "tags" in element:
        lat = element["lat"]
        lon = element["lon"]
        religious_locations.append((lon, lat))
        religious_name = element["tags"].get("name", "")
        religious_type = element["tags"].get("amenity", "")
        religious_types.append(religious_type)

# create a GeoDataFrame from the religious facility locations and types
religious_facilities_gdf = gpd.GeoDataFrame(
    {"religious_type": religious_types},
    geometry=gpd.points_from_xy([lon for lon, lat in religious_locations], [lat for lon, lat in religious_locations])
)

# set the coordinate reference system (CRS) if known
religious_facilities_gdf.crs = "EPSG:4326"  # WGS84 coordinate system
output_shapefile = f'{country}_religious_facilities/religious_facilities.shp'

# save the GeoDataFrame to a shapefile
religious_facilities_gdf.to_file(output_shapefile)

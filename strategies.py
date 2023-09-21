import geopandas as gpd
from geopandas import GeoDataFrame

# Your dictionary
data_dict = {
    "type": "FeatureCollection",
    "crs": { "type": "name", "properties": { "name": "urn:ogc:def:crs:EPSG::32619" } },
    "features": [
        { "type": "Feature", "properties": { }, "geometry": { "type": "Polygon", "coordinates": [ [ [ 317000.0, 4883000.0 ], [ 317000.0, 4884000.0 ], [ 316000.0, 4884000.0 ], [ 316000.0, 4883000.0 ], [ 317000.0, 4883000.0 ] ] ] } }
    ]
}

# Convert dictionary to GeoDataFrame
gdf = GeoDataFrame.from_features(data_dict["features"], crs="EPSG:32619")

# If you still want to buffer (even with 0 distance), you can do:
gdf["geometry"] = gdf.buffer(0)

# Reproject to Web Mercator (EPSG:3857)
gdf_web_mercator = gdf.to_crs(epsg=4326)

# Save to GeoJSON
gdf_web_mercator.to_file("output_file.geojson", driver='GeoJSON')


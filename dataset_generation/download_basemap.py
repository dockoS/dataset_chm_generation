from common import stack_masks,mask_to_ceo_format,create_mask,get_one_satelite_image,filtering_tree_by_area ,\
    coordinate_to_bbox,get_index_last_repertory,partition,griding_polygon,tms_to_geotiff,compute_area,save_mask_png,\
        add_physic_parameter_to_df,create_vrt,get_statistics,project_coordinates_4326_3857,filter_overlapping_polygons,create_tiff,create_tree_masks,create_vrt_parallel
import click
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
#import cv2
import os
from loguru import logger
import json
from shapely.geometry import Polygon,LineString,MultiPolygon,box
import geojson
from datetime import datetime,timedelta
from PIL import Image
from shapely.geometry import Polygon,LineString,MultiPolygon
import pandas as pd
from joblib import Parallel,delayed
def create_tiff_files_parallel(geojson_path:str,output_path:str="../dataset"):
    os.makedirs(output_path,exist_ok=True)
    if not  geojson_path.endswith("geojson"):
        raise ValueError(f"{geojson_path} must be a geojson file")
    with open(geojson_path) as f:
        data = geojson.load(f)
    try:
        feature=data["features"][0]
        coordinates=feature["geometry"]["coordinates"][0]
    except:
        raise ValueError("please check the attribut coordinates")
    try:
        rep_name=feature["name"]
        if rep_name.startswith("image"):
            raise ValueError("change the value of the attribut name. To avoid ambiguty it have not to begins with image")
    except:
            current_index_rep=get_index_last_repertory(output_path)
            rep_name=f"image{current_index_rep+1}"            
    rep_path=os.path.join(output_path,rep_name)
    os.makedirs(rep_path,exist_ok=True)
    grids=griding_polygon(coordinates)
    
    print(f"number of polygon{len(grids)}")
    filenames=Parallel(n_jobs=-1)(delayed(create_tiff)(grids[i],os.path.join(rep_path,f'image{i}.tiff')) for i in range(len(grids)))
    return coordinates,filenames

def create_tiff_files_parallel_modify(geojson_path:str,output_path:str="../dataset"):
    os.makedirs(output_path,exist_ok=True)
    if not  geojson_path.endswith("geojson"):
        raise ValueError(f"{geojson_path} must be a geojson file")
    with open(geojson_path) as f:
        data = geojson.load(f)
    try:
        features=data["features"]
        
    except:
        raise ValueError("please check the attribut coordinates")
    # try:
    #     rep_name=feature["name"]
    #     if rep_name.startswith("image"):
    #         raise ValueError("change the value of the attribut name. To avoid ambiguty it have not to begins with image")
    # except:
    #         current_index_rep=get_index_last_repertory(output_path)
    #         rep_name=f"image{current_index_rep+1}"            
    # rep_path=os.path.join(output_path,rep_name)
    # os.makedirs(rep_path,exist_ok=True)
    index_pol=0
    for feature in features:
        print("yes")
        coordinates=feature["geometry"]["coordinates"][0]
        grids=griding_polygon(coordinates)
      
      
        
    
        logger.success(f"number of polygon{len(grids)}")
        filenames=Parallel(n_jobs=-1)(delayed(create_tiff)(grids[i],os.path.join(output_path,f'{index_pol}_{i}.tiff')) for i in range(len(grids)))
        index_pol+=1
if __name__ == "__main__":
    create_tiff_files_parallel_modify("../geojson_path/hres.geojson","../hrdataset")
    
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
from rasterio import MemoryFile,Affine

import requests
import os
import rasterio
from shapely.geometry import box
import geopandas as gpd

def process_url(url,output_chm_path,output_geojson_path):
    # Define the local file path
    name_geojson=url.split("/")[-1].split(".")[0]+".geojson"
    name_chm_file=url.split("/")[-1]
    geojson_path = os.path.join(output_geojson_path,name_geojson)  # store with the same filename in the current directory
    chm_file_path=os.path.join(output_chm_path,name_chm_file)
    # Download the file
    response = requests.get(url)
    with open(chm_file_path, 'wb') as f:
        f.write(response.content)

    # Read the bounding box with rasterio
    with rasterio.open(chm_file_path) as src:
        bbox = src.bounds
        crs=src.crs
    # Convert the bounding box to a shapely polygon
    polygon = box(*bbox)

    # Create a geodataframe and save as GeoJSON
    gdf = gpd.GeoDataFrame({'geometry': [polygon]}, crs=crs)
    #geojson_path = os.path.splitext(local_path)[0] + ".geojson"
    gdf.to_file(geojson_path, driver="GeoJSON")

# List of URLs
def read_url_download(data_path="../geojson_path/NEON_training_images.csv",output_geojson_path="../geojson_path/chm",output_chm_path="../chmdata"):
    try:
        data=pd.read_csv(data_path)
    except:
        raise ("give a csv file")
    urls=list(data["url"])
    Parallel(n_jobs=-1)(delayed(process_url)(url,output_chm_path,output_geojson_path) for url in urls)
def get_box_4326(filename,output_path="../geojson_path/chm4326",rgb_path="../RGBdata"):
    gdf = gpd.read_file(filename)
    new_name=os.path.join(output_path,filename.split("/")[-1].split(".")[0])+".geojson"
    tiff_rgb_filename=os.path.join(output_path,filename.split("/")[-1].split(".")[0])+".tiff"
    gdf["geometry"] = gdf.buffer(0)

    # Reproject to Web Mercator (EPSG:4326)
    gdf_web_mercator = gdf.to_crs(epsg=4326)
    polygon_coords = list(gdf_web_mercator.geometry.iloc[0].exterior.coords)
    create_tiff(Polygon(polygon_coords),os.path.join(tiff_rgb_filename,rgb_path))
    # Create a geodataframe and save as GeoJSON
    #gdf = gpd.GeoDataFrame({'geometry': [polygon]}, crs=crs)
    #geojson_path = os.path.splitext(local_path)[0] + ".geojson"
    gdf_web_mercator.to_file(new_name, driver="GeoJSON")
            
def get_all_bbox(geojson_folder_path="../geojson_path/chm",output_path="../geojson_path/chm4326",rgb_path="../RGBdata"):
    filenames=os.listdir(geojson_folder_path)
    Parallel(n_jobs=-1)(delayed(get_box_4326)(os.path.join(geojson_folder_path,filename),output_path,rgb_path) for filename in filenames)           
#def download_tiff(geojson_output):
import os
import rasterio
from PIL import Image
import numpy as np
def resize_raster_image(input_folder, output_folder, desired_size=(1000, 1000)):
    # Ensure the output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".tif"):  # Check if the file is a TIFF
            filepath = os.path.join(input_folder, filename)
            
            with rasterio.open(filepath) as src:
                # Read the image bands - assuming the image has 3 bands corresponding to RGB
                red, green, blue = src.read()

                # Stack bands to create an RGB image
                rgb = np.stack((red, green, blue), axis=-1)

                # Convert to PIL Image for easy resizing
                img = Image.fromarray(rgb.astype('uint8'))

                # Resize the image
                img_resized = img.resize(desired_size)

                # Save the resized image
                img_resized.save(os.path.join(output_folder, filename))
def save_single_band_raster_in_jpeg(filename,output_folder):
             
    with rasterio.open(filename) as src:
            # Read the single band from the raster
            band = src.read(1)

            # Convert to PIL Image for easy resizing
            img = Image.fromarray(band.astype('uint8'), 'L')  # 'L' mode is for grayscale

            # Resize the image
            #img_resized = img.resize(desired_size)

            # Save the resized image
            img.save(os.path.join(output_folder, filename.split(".")[0]+'.jpeg'))

def resize_single_band_raster(input_folder, output_folder):
    # Ensure the output directory exists
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    filenames=os.listdir(input_folder)
    Parallel(n_jobs=-1)(delayed(save_single_band_raster_in_jpeg)(os.path.join(input_folder, filename),output_folder) for filename in filenames)           

def crop_and_save_images(input_folder, output_folder, crop_size=(224, 224), stride=112):
    # Ensure the output directory exists
    print("yes")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpeg"):  # Check if the file is a JPEG image
     
            filepath = os.path.join(input_folder, filename)
            
            with Image.open(filepath) as img:
                width, height = img.size
                
                # Calculate the number of rows and columns for cropping
                rows = (height - crop_size[1]) // stride + 1
                cols = (width - crop_size[0]) // stride + 1
                
                # Iterate through rows and columns to crop and save
                for i in range(rows):
                    for j in range(cols):
                        left = j * stride
                        upper = i * stride
                        right = left + crop_size[0]
                        lower = upper + crop_size[1]
                        
                        # Crop the image
                        cropped_img = img.crop((left, upper, right, lower))
                        
                        # Save the cropped image with a unique name
                        crop_filename = f"{filename.replace('.jpg', '')}_crop_{i}_{j}.jpg"
                        crop_filepath = os.path.join(output_folder, crop_filename)
                        cropped_img.save(crop_filepath)

crop_and_save_images('../chmjpeg', '../chmjpegCrop')
#resize_single_band_raster('../chmdata', '../chmjpeg')

#get_all_bbox()


import cv2
import numpy as np
from skimage.measure import find_contours
import rasterio
from rasterio.enums import Resampling
from PIL import Image 
from rasterio import MemoryFile,Affine
import os
from contextlib import contextmanager  
import math
from loguru import logger
import pandas as pd
import matplotlib.pyplot as plt
import urllib
from rasterio import warp
from shapely.geometry import Polygon,Point,box
import requests
from typing import List,Dict
from datetime import datetime, timedelta
import binascii
import pycrs
import pandas as pd
import pyproj
from tqdm import tqdm
from pyproj import transform  as projTrans
import geopandas as gpd
from shapely.geometry import Polygon,LineString,MultiPolygon,box
from pyproj import CRS
import os
from pyproj import Proj,itransform
from shapely.ops import transform
import shapely
#from segment_anything import sam_model_registry, SamAutomaticMaskGenerator,SamPredictor
import matplotlib
from joblib import Parallel,delayed
from rasterio.mask import mask
from shapely.strtree import STRtree

def coordinate_to_bbox(coordinates:List):
    lon=[pt[0] for pt in coordinates ]
    lat=[pt[1] for pt in coordinates ]
    return [min(lon),min(lat),max(lon),max(lat)]

def check_file_path(file_path, make_dirs=True):
    """Gets the absolute file path.
    Args:
        file_path (str): The path to the file.
        make_dirs (bool, optional): Whether to create the directory if it does not exist. Defaults to True.
    Raises:
        FileNotFoundError: If the directory could not be found.
        TypeError: If the input directory path is not a string.
    Returns:
        str: The absolute path to the file.
    """
    if isinstance(file_path, str):
        if file_path.startswith("~"):
            file_path = os.path.expanduser(file_path)
        else:
            file_path = os.path.abspath(file_path)

        file_dir = os.path.dirname(file_path)
        if not os.path.exists(file_dir) and make_dirs:
            os.makedirs(file_dir)

        return file_path

    else:
        raise TypeError("The provided file path must be a string.")


def temp_file_path(extension):
    """Returns a temporary file path.

    Args:
        extension (str): The file extension.

    Returns:
        str: The temporary file path.
    """

    import tempfile
    import uuid

    if not extension.startswith("."):
        extension = "." + extension
    file_id = str(uuid.uuid4())
    file_path = os.path.join(tempfile.gettempdir(), f"{file_id}{extension}")

    return file_path
def image_to_cog(source, dst_path=None, profile="deflate", **kwargs):
    """Converts an image to a COG file.

    Args:
        source (str): A dataset path, URL or rasterio.io.DatasetReader object.
        dst_path (str, optional): An output dataset path or or PathLike object. Defaults to None.
        profile (str, optional): COG profile. More at https://cogeotiff.github.io/rio-cogeo/profile. Defaults to "deflate".

    Raises:
        ImportError: If rio-cogeo is not installed.
        FileNotFoundError: If the source file could not be found.
    """
    try:
        from rio_cogeo.cogeo import cog_translate
        from rio_cogeo.profiles import cog_profiles

    except ImportError:
        raise ImportError(
            "The rio-cogeo package is not installed. Please install it with `pip install rio-cogeo` or `conda install rio-cogeo -c conda-forge`."
        )

    if not source.startswith("http"):
        source = check_file_path(source)

        if not os.path.exists(source):
            raise FileNotFoundError("The provided input file could not be found.")

    if dst_path is None:
        if not source.startswith("http"):
            dst_path = os.path.splitext(source)[0] + "_cog.tif"
        else:
            dst_path = temp_file_path(extension=".tif")

    dst_path = check_file_path(dst_path)

    dst_profile = cog_profiles.get(profile)
    cog_translate(source, dst_path, dst_profile, **kwargs)
def tms_to_geotiff(
    output,
    bbox,
    zoom=None,
    resolution=None,
    source="OpenStreetMap",
    to_cog=False,
    return_image=False,
    overwrite=False,
    quiet=False,
    **kwargs,
):
    """Download TMS tiles and convert them to a GeoTIFF. The source is adapted from https://github.com/gumblex/tms2geotiff.
        Credits to the GitHub user @gumblex.

    Args:
        output (str): The output GeoTIFF file.
        bbox (list): The bounding box [minx, miny, maxx, maxy], e.g., [-122.5216, 37.733, -122.3661, 37.8095]
        zoom (int, optional): The map zoom level. Defaults to None.
        resolution (float, optional): The resolution in meters. Defaults to None.
        source (str, optional): The tile source. It can be one of the following: "OPENSTREETMAP", "ROADMAP",
            "SATELLITE", "TERRAIN", "HYBRID", or an HTTP URL. Defaults to "OpenStreetMap".
        to_cog (bool, optional): Convert to Cloud Optimized GeoTIFF. Defaults to False.
        return_image (bool, optional): Return the image as PIL.Image. Defaults to False.
        overwrite (bool, optional): Overwrite the output file if it already exists. Defaults to False.
        quiet (bool, optional): Suppress output. Defaults to False.
        **kwargs: Additional arguments to pass to gdal.GetDriverByName("GTiff").Create().

    """

    import os
    import io
    import math
    import itertools
    import concurrent.futures

    import numpy
    from PIL import Image

    try:
        from osgeo import gdal, osr
    except ImportError:
        raise ImportError("GDAL is not installed. Install it with pip install GDAL")

    try:
        import httpx

        SESSION = httpx.Client()
    except ImportError:
        import requests

        SESSION = requests.Session()

    if not overwrite and os.path.exists(output):
        print(f"The output file {output} already exists. Use `overwrite=True` to overwrite it.")
        return

    xyz_tiles = {
        "OPENSTREETMAP": {
            "url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
            "attribution": "OpenStreetMap",
            "name": "OpenStreetMap",
        },
        "ROADMAP": {
            "url": "https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
            "attribution": "Google",
            "name": "Google Maps",
        },
        "SATELLITE": {
            "url": "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
            "attribution": "Google",
            "name": "Google Satellite",
        },
        "TERRAIN": {
            "url": "https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}",
            "attribution": "Google",
            "name": "Google Terrain",
        },
        "HYBRID": {
            "url": "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
            "attribution": "Google",
            "name": "Google Satellite",
        },
    }

    if isinstance(source, str) and source.upper() in xyz_tiles:
        source = xyz_tiles[source.upper()]["url"]
    elif isinstance(source, str) and source.startswith("http"):
        pass
    else:
        raise ValueError(
            'source must be one of "OpenStreetMap", "ROADMAP", "SATELLITE", "TERRAIN", "HYBRID", or a URL'
        )

    def resolution_to_zoom_level(resolution):
        """
        Convert map resolution in meters to zoom level for Web Mercator (EPSG:3857) tiles.
        """
        # Web Mercator tile size in meters at zoom level 0
        initial_resolution = 156543.03392804097

        # Calculate the zoom level
        zoom_level = math.log2(initial_resolution / resolution)

        return int(zoom_level)

    if isinstance(bbox, list) and len(bbox) == 4:
        west, south, east, north = bbox
    else:
        raise ValueError(
            "bbox must be a list of 4 coordinates in the format of [xmin, ymin, xmax, ymax]"
        )

    if zoom is None and resolution is None:
        raise ValueError("Either zoom or resolution must be provided")
    elif zoom is not None and resolution is not None:
        raise ValueError("Only one of zoom or resolution can be provided")

    if resolution is not None:
        zoom = resolution_to_zoom_level(resolution)

    EARTH_EQUATORIAL_RADIUS = 6378137.0

    Image.MAX_IMAGE_PIXELS = None

    gdal.UseExceptions()
    web_mercator = osr.SpatialReference()
    web_mercator.ImportFromEPSG(3857)

    WKT_3857 = web_mercator.ExportToWkt()

    def from4326_to3857(lat, lon):
        xtile = math.radians(lon) * EARTH_EQUATORIAL_RADIUS
        ytile = (
            math.log(math.tan(math.radians(45 + lat / 2.0))) * EARTH_EQUATORIAL_RADIUS
        )
        return (xtile, ytile)

    def deg2num(lat, lon, zoom):
        lat_r = math.radians(lat)
        n = 2**zoom
        xtile = (lon + 180) / 360 * n
        ytile = (1 - math.log(math.tan(lat_r) + 1 / math.cos(lat_r)) / math.pi) / 2 * n
        return (xtile, ytile)

    def is_empty(im):
        extrema = im.getextrema()
        if len(extrema) >= 3:
            if len(extrema) > 3 and extrema[-1] == (0, 0):
                return True
            for ext in extrema[:3]:
                if ext != (0, 0):
                    return False
            return True
        else:
            return extrema[0] == (0, 0)

    def paste_tile(bigim, base_size, tile, corner_xy, bbox):
        if tile is None:
            return bigim
        im = Image.open(io.BytesIO(tile))
        mode = "RGB" if im.mode == "RGB" else "RGBA"
        size = im.size
        if bigim is None:
            base_size[0] = size[0]
            base_size[1] = size[1]
            newim = Image.new(
                mode, (size[0] * (bbox[2] - bbox[0]), size[1] * (bbox[3] - bbox[1]))
            )
        else:
            newim = bigim

        dx = abs(corner_xy[0] - bbox[0])
        dy = abs(corner_xy[1] - bbox[1])
        xy0 = (size[0] * dx, size[1] * dy)
        if mode == "RGB":
            newim.paste(im, xy0)
        else:
            if im.mode != mode:
                im = im.convert(mode)
            if not is_empty(im):
                newim.paste(im, xy0)
        im.close()
        return newim

    def finish_picture(bigim, base_size, bbox, x0, y0, x1, y1):
        xfrac = x0 - bbox[0]
        yfrac = y0 - bbox[1]
        x2 = round(base_size[0] * xfrac)
        y2 = round(base_size[1] * yfrac)
        imgw = round(base_size[0] * (x1 - x0))
        imgh = round(base_size[1] * (y1 - y0))
        retim = bigim.crop((x2, y2, x2 + imgw, y2 + imgh))
        if retim.mode == "RGBA" and retim.getextrema()[3] == (255, 255):
            retim = retim.convert("RGB")
        bigim.close()
        return retim

    def get_tile(url):
        retry = 3
        while 1:
            try:
                r = SESSION.get(url, timeout=60)
                break
            except Exception:
                retry -= 1
                if not retry:
                    raise
        if r.status_code == 404:
            return None
        elif not r.content:
            return None
        r.raise_for_status()
        return r.content

    def draw_tile(
        source, lat0, lon0, lat1, lon1, zoom, filename, quiet=False, **kwargs
    ):
        x0, y0 = deg2num(lat0, lon0, zoom)
        x1, y1 = deg2num(lat1, lon1, zoom)
        if x0 > x1:
            x0, x1 = x1, x0
        if y0 > y1:
            y0, y1 = y1, y0
        corners = tuple(
            itertools.product(
                range(math.floor(x0), math.ceil(x1)),
                range(math.floor(y0), math.ceil(y1)),
            )
        )
        totalnum = len(corners)
        futures = []
        with concurrent.futures.ThreadPoolExecutor(5) as executor:
            for x, y in corners:
                futures.append(
                    executor.submit(get_tile, source.format(z=zoom, x=x, y=y))
                )
            bbox = (math.floor(x0), math.floor(y0), math.ceil(x1), math.ceil(y1))
            bigim = None
            base_size = [256, 256]
            for k, (fut, corner_xy) in enumerate(zip(futures, corners), 1):
                bigim = paste_tile(bigim, base_size, fut.result(), corner_xy, bbox)
                if not quiet:
                    print("Downloaded image %d/%d" % (k, totalnum))

        if not quiet:
            print("Saving GeoTIFF. Please wait...")
        img = finish_picture(bigim, base_size, bbox, x0, y0, x1, y1)
        imgbands = len(img.getbands())
        driver = gdal.GetDriverByName("GTiff")

        if "options" not in kwargs:
            kwargs["options"] = [
                "COMPRESS=DEFLATE",
                "PREDICTOR=2",
                "ZLEVEL=9",
                "TILED=YES",
            ]

        gtiff = driver.Create(
            filename,
            img.size[0],
            img.size[1],
            imgbands,
            gdal.GDT_Byte,
            **kwargs,
        )
        xp0, yp0 = from4326_to3857(lat0, lon0)
        xp1, yp1 = from4326_to3857(lat1, lon1)
        pwidth = abs(xp1 - xp0) / img.size[0]
        pheight = abs(yp1 - yp0) / img.size[1]
        gtiff.SetGeoTransform((min(xp0, xp1), pwidth, 0, max(yp0, yp1), 0, -pheight))
        gtiff.SetProjection(WKT_3857)
        for band in range(imgbands):
            array = numpy.array(img.getdata(band), dtype="u8")
            array = array.reshape((img.size[1], img.size[0]))
            band = gtiff.GetRasterBand(band + 1)
            band.WriteArray(array)
        gtiff.FlushCache()

        if not quiet:
            print(f"Image saved to {filename}")
        return img

    try:
        image = draw_tile(
            source, south, west, north, east, zoom, output, quiet, **kwargs
        )
        if return_image:
            return image
        if to_cog:
            image_to_cog(output, output)
    except Exception as e:
        raise Exception(e)

# def get_mask_generator(checkpoint_path,model_type="default",device="cpu"):

#     sam=sam_model_registry[model_type](checkpoint=checkpoint_path)
#     sam.to(device=device)  
#     return SamAutomaticMaskGenerator(sam)

def get_index_last_repertory(data_path:str):
    rep_names=os.listdir(data_path)
    rep_names=[rep for rep in rep_names if rep.startswith("image")]
    if len(rep_names)==0:
        return 0
    rep_names.sort()

    last_rep=rep_names[-1]
    index= ''.join(list(filter(str.isdigit,last_rep )))
    if index=='':
        return 0
    return int(index)



def compute_area(polygon,mercator=True):
    if mercator:
        return polygon.area
    wgs84 = pyproj.CRS('EPSG:4326')
    utm = pyproj.CRS('EPSG:3857')
    project = pyproj.Transformer.from_crs(wgs84, utm, always_xy=True).transform
    projected_area = transform(project, polygon).area
    return projected_area

from shapely.geometry import Polygon
import numpy as np

def grid_bounds(geom):
    minx, miny, maxx, maxy = geom.bounds
    delta=0.1
    
    grid = []
    print((maxx - minx))
    print((maxy - miny))
    print(compute_area(geom))
    if compute_area(geom)<=50000:
        return [geom]

    while len(grid)==0:
        print(compute_area(geom))
        
        nx = int((maxx - minx)*1000/delta)
        ny = int((maxy - miny)*1000/delta)
        
        print(nx,ny)
        gx, gy = np.linspace(minx,maxx,nx), np.linspace(miny,maxy,ny)
        print(gx,gy)
        if len(gx)>=2  and len(gy)>=2:
            
            poly_ij = Polygon([[gx[0],gy[0]],[gx[0],gy[1]],[gx[1],gy[1]],[gx[1],gy[0]]])

            print(compute_area(poly_ij))
            if  compute_area(poly_ij)<50000 or  compute_area(poly_ij)>52000:
                delta+=0.01
                continue
        else:
            continue
            
        for i in range(len(gx)-1):
            for j in range(len(gy)-1):
                poly_ij = Polygon([[gx[i],gy[j]],[gx[i],gy[j+1]],[gx[i+1],gy[j+1]],[gx[i+1],gy[j]]])

                grid.append( poly_ij )
    return grid

from shapely.prepared import prep

def partition(coordinates):
    geom=Polygon(coordinates)
    prepared_geom = prep(geom)
    grid = list(filter(prepared_geom.intersects, grid_bounds(geom)))
    return grid

          


def griding_polygon(coordinates):
    
    geom=Polygon(coordinates)
    xmin, ymin, xmax, ymax= geom.bounds

    ## real cell dimensions
    #5hect  re cell
    cell_width  = 0.00214/1.5
    cell_height = 0.00199/1.5

    grid_cells = []
    for x0 in np.arange(xmin, xmax+cell_width, cell_width ):
        for y0 in np.arange(ymin, ymax+cell_height, cell_height):
            x1 = x0+cell_width
            y1 = y0+cell_height
            new_cell = shapely.geometry.box(x0, y0, x1, y1)
            if new_cell.intersects(geom):
                #print(new_cell)
                #print(compute_area(Polygon(new_cell))//10000)
                grid_cells.append(Polygon(new_cell))
            else:
                pass
    return grid_cells


def get_dates(start_date,end_date):
    start_date= datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    date_list = []
    while start_date <= end_date:
        date_list.append(start_date.strftime("%Y-%m-%d"))
        start_date += timedelta(days=1)

    return date_list
def project_pt_4326_3857(input_longitude,input_latitude):
    # Define the input and output coordinate systems
    input_crs = Proj('EPSG:4326')
    output_crs = Proj('EPSG:3857')
    # Project the coordinates to Web Mercator (EPSG 3857)
    output_longitude, output_latitude = projTrans(input_crs, output_crs, input_longitude, input_latitude)
    return [output_longitude,output_latitude]
def project_coordinates_4326_3857(coordinates):
    return [project_pt_4326_3857(*pt) for pt  in coordinates]
def transform_bbox(bbox3857):
    # Define the input and output coordinate reference systems
    in_crs = pyproj.CRS('EPSG:3857')
    out_crs = pyproj.CRS('EPSG:4326')
    
    # Create a transformer to convert from CRS3857 to CRS4326
    transformer = pyproj.Transformer.from_crs(in_crs, out_crs)
    
    # Transform the bbox to CRS4326
    miny, minx, maxy, maxx = bbox3857
    bbox4326 = transformer.transform_bounds(minx, miny, maxx, maxy)
    return list(bbox4326)
def get_contours_in_crs_4326(arr,dst_transform):
    print(type(arr))
    # Assuming the mask is stored in a numpy array called "mask"
    contours = find_contours(arr, 0.5)
    pol=[]

    # Iterate over all contours found
    for contour in contours:
        # Each contour is a list of (y, x) coordinates, so we need to transpose it
        # and subtract 0.5 to get the correct pixel coordinates
        y = contour[:, 0] - 0.5
        x = contour[:, 1] - 0.5
        x=[math.ceil(i) for i in x]
        y=[math.ceil(i) for i in y]
        
        x, y = rasterio.transform.xy(dst_transform, x, y)
        
        # Print the coordinates of the contour
        pol.extend(list(map(list, zip(x, y))))
    return pol 
def find_closest_number_left(num, lst):
    
    # Base case: if the list has one or zero elements, return the list
    if len(lst) == 1:

        return lst[0], 0

    if len(lst)==0:
        return None,None


    # Initialize the left and right indices
    left_idx = 0
    right_idx = len(lst) - 1

    # Check if the list is in ascending or descending order
    is_descending = lst[left_idx] > lst[right_idx]

    # Loop until the left and right indices are adjacent
    while right_idx - left_idx > 1:
        # Calculate the middle index
        mid_idx = (left_idx + right_idx) // 2

        # Determine the appropriate comparison operator based on the order of the list
        if is_descending:
            if lst[mid_idx] < num:
                right_idx = mid_idx
            else:
                left_idx = mid_idx
        else:
            if lst[mid_idx] > num:
                right_idx = mid_idx
            else:
                left_idx = mid_idx


    # Return the closest number between the two adjacent indices
    left_closest = lst[left_idx]
    right_closest = lst[right_idx]
    return (left_closest, left_idx) if abs(num - left_closest) < abs(num - right_closest) else (right_closest, right_idx)


def get_lon_lat_raster(raster_data):
    dataset=raster_data
    transform = dataset.transform
    #Sable code
# Get the width and height of the raster
    width = dataset.width
    height = dataset.height
    rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    # Convert the row and column indices to x and y coordinates in the CRS of the raster dataset
    x, y = rasterio.transform.xy(transform, rows.flatten(), cols.flatten())
    lon, lat =x,y
    lon = np.asarray(x).reshape((height, width))[0]
    lat = np.asarray(y).reshape((height, width))[:,0]
    return lon,lat
def get_raster_projected(dataset,proj="EPSG:4326"):
    
# Read the raster data and extract the necessary information
    profile = dataset.profile
    crs = dataset.crs
# Define the target CRS for the reprojection
    dst_crs = 'EPSG:4326'
    # Calculate the transform for the reprojection
    dst_transform, _, _ = warp.calculate_default_transform(
        crs, dst_crs, profile['width'], profile['height'], *dataset.bounds)
    return dst_transform
        

    
def get_contours_in_crs(arr,lon,lat):
    # Assuming the mask is stored in a numpy array called "mask"
    contours = find_contours(arr, 0.5)
    pol=[]

    # Iterate over all contours found
    for contour in contours:
        # Each contour is a list of (y, x) coordinates, so we need to transpose it
        # and subtract 0.5 to get the correct pixel coordinates
        y = contour[:, 0] - 0.5
        x = contour[:, 1] - 0.5
        x=[math.ceil(i) for i in x]
        y=[math.ceil(i) for i in y]
        
        x, y = lon[x],lat[y]
        # Print the coordinates of the contour
        pol.extend(list(map(list, zip(x,y))))
    return Polygon(pol)

def show_anns(anns):
    if len(anns)==0:
        return 
    #sorted_anns=sorted(anns,key=(lambda x:x['area']),reverse=True)
    ax=plt.gca()
    ax.set_autoscale_on(False)
    for  mask in anns:
        img=np.ones((mask.shape[0],mask.shape[1],3))
        color_mask=np.random.random((1,3)).tolist()[0]
        for i in range(3):
            img[:,:,i]=color_mask[i]
        ax.imshow(np.dstack((img,mask*0.35)))
def save_mask_png(img,segmentation,filename):
    print(f'png name  {filename}')
    fig=plt.figure(figsize=(20,20))
    plt.imshow(img)
    show_anns(segmentation)
    plt.show()
    plt.savefig(filename)
    print("image saved")
    
def create_mask(mask_generator,raster_path):
    with rasterio.open(raster_path) as dataset:
        img=dataset.read()
        r,g,b=img
        print("1")
        img=np.stack([r,g,b],axis=2)
        lon,lat=get_lon_lat_raster(dataset)
        print("2")
        masks=mask_generator.generate(img)
        masks_pol=list(map(lambda mask: get_contours_in_crs(mask['segmentation'],lon,lat),masks))
        segmentation=list(map(lambda  mask: mask["segmentation"],masks))
        print("3")
    #get only trees .Postula: Tree area do not be greater than 340 square meters zoom level 21 =26 centimeter/pixel
        return segmentation,masks_pol,dataset.bounds,img
    
def create_tree_masks(mask_generator,raster_image_path,min_tree_area,max_tree_area):
    segmentation,masks,_,img=create_mask(mask_generator,raster_image_path)
    masks=filtering_tree_by_area(masks,min_tree_area,max_tree_area)
    non_overlap_masks=filter_overlapping_polygons(masks)    
    #save_mask_png(img,segmentation,f'{raster_image_path}.png')
    return non_overlap_masks
def get_one_satelite_image(area,bbox,start_date="2023-02-01",end_date="2023-04-01",satellite_id="s2_l2a"):
    wcs_url = 'https://ows.digitalearth.africa'
    for _date in tqdm(get_dates(start_date,end_date),desc=f'searching images for satelites= {satellite_id}'):
        image_format = urllib.parse.quote_plus('image/geotiff')
        long = urllib.parse.quote_plus(f'{bbox[0]},{bbox[2]}')
        lat = urllib.parse.quote_plus(f'{bbox[1]},{bbox[3]}')
        sat_id=urllib.parse.quote(satellite_id)
        cloud_cover_percentage=2
        scale_factor=1
        if area==False:
            scale_factor = 30
        else:   
            scale_factor = 1
            if area <= 3:
                scale_factor = 20
            elif area <= 250:
                scale_factor = 8
            elif area <= 500:
                scale_factor = 4
        
        params = f'service=WCS&request=GetCoverage&version=2.0.0&coverageId={sat_id}&format={image_format}' \
                f'&subset=Long({long})&subset=Lat({lat})&subset=time("{_date}")&subsettingCrs=EPSG:4326' \
                f'&outputCrs=EPSG:3857&scaleaxes=x({scale_factor}),y({scale_factor})&cloud_cover={cloud_cover_percentage}'
     
        resp = requests.get(f'{wcs_url}?{params}')
        if resp.status_code==404:
            pass
        else:
            resp.raise_for_status()
            data = binascii.b2a_base64(resp.content).decode('utf-8')
            data = binascii.a2b_base64(data)
            return data    
def compute_entropy(r,g,b):
    
    from skimage.feature import graycomatrix, graycoprops
    img=np.stack([r,g,b],axis=2)
    gray_img=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_normalized = cv2.normalize(gray_img, None, 0, 255,
        cv2.NORM_MINMAX, dtype=cv2.CV_32SC1)


    glcm = graycomatrix(img_normalized, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

# calculate the entropy of the GLCM
    entropy = -np.sum(glcm * np.log2(glcm + (glcm == 0)))
    energy = graycoprops(glcm, 'energy')[0, 0]
    return  energy
def filtering_tree_by_area(masks:List,min_tree_area=10,max_tree_area=300):
    tree_masks=[]
    for mask in masks:
        P = mask
        area_p=compute_area(P)
        logger.error(f'area {area_p}')
        if mask.is_valid:
            if area_p>min_tree_area and area_p<max_tree_area:
                tree_masks.append(mask)       
    return tree_masks

def stack_masks(mask_generator,all_img_path):
    filenames=[file for file in os.listdir(all_img_path) if file.endswith(".png")]
    masks=[]
    for file in filenames:
        png_path=os.path.join(all_img_path,file)
        raster_path=os.path.join(all_img_path,f'{file.split(".")[0]}.tiff')
        print(png_path)
        masks.extend(create_mask(mask_generator,png_path,raster_path))
        break
    return masks
    
    
def mask_to_ceo_format(mask,csv_path,name_csv):
    df=pd.DataFrame(mask)
    path=os.path.join(csv_path,name_csv)
    df.to_csv(f'{path}.csv')
    return df


def polygon_to_circle(polygon):
    
        # Find the bounding box of the polygon
        minx, miny, maxx, maxy = polygon.bounds

        # Find the center of the bounding box
        center_x = (minx + maxx) / 2
        center_y = (miny + maxy) / 2
        center = Point(center_x, center_y)

        # Calculate the distance from the center to any vertex of the polygon

        radius = center.distance(Point(polygon.exterior.coords[0]))

        # Create a circle with the center and radius
        circle = center.buffer(radius)
        return circle

def filter_overlapping_polygons(polygons):
    index = STRtree(polygons)
    filtered_polygons = []
    for polygon in polygons:
        overlaps = index.query(polygon)
        print(overlaps)
        is_largest = True
        for overlap in overlaps:
            if polygon != polygons[overlap] and polygon.intersects(polygons[overlap]):
                if polygon.area < polygons[overlap].area:
                    is_largest = False
                    break
        if is_largest:
            filtered_polygons.append(polygon)
    return filtered_polygons


def compute_physic_parameter(polygon,type):
    
    # if not str_geometry.startswith("POLYGON"):
    #     return -1
    #         # Calculate the area of the polygon in square centimeters

    # polygon=shapely.wkt.loads(str_geometry)
    circle=polygon_to_circle(polygon)
    #crown_area_in_cm2=polygon.transform(6933, clone=True).area
    crown_area_in_cm2=compute_area(polygon)
    if crown_area_in_cm2==0.0:
        carbon=0
        return carbon

    
    CD=2*math.sqrt((crown_area_in_cm2/math.pi))
    #logger.info(f"crown size=={crown_area_in_cm2}")
    #logger.success(f"CD= {CD}")
    #Compute DBH and AGB(Kg) for savana 
    DBH=(math.exp(1.154+(1.248*math.log(CD*1.27)))*((math.exp((0.3315)**2)/2)))
    #DBH= 0.27+ CD +24
    AGB=0.091*(DBH**2.472)
    #AGB en tonnes
    if type=="biomass":
        AGB=AGB/1000
        return AGB*0.3
    elif type=="carbon":
        
        #compute BGB(Kg) with mbaye's method
        BGB=(AGB*0.26)+24 
        above_ground_carbon=AGB*0.47
        CO2=(AGB+BGB)*0.5
        
        #logger.error(f"agc=={above_ground_carbon}")
        carbon=CO2/0.2727
        
        return carbon   
    else:
        raise ValueError("Type not  valid")
    #return AGB*0.3
    
    
    
    
    

def add_physic_parameter_to_df(df_trees,type="carbon"):
 

    try:
        str_polygons=list(df_trees["sample_geom"])
        print(df_trees)
    except:
        raise ValueError("Verify the polygon attribut, the correct name is sample_geom -or the class_val_feat or check the file path or the class_name_feat")
    tqdm.pandas(desc="compute carbon") 
    df_trees[f"{type}"] = df_trees.progress_apply(
            lambda row:compute_physic_parameter(row["sample_geom"],type),
            axis=1
        )    
    result=df_trees[df_trees[f"{type}"]!=-1]
    #df_trees.to_csv("touba toul.csv")
    return result[["sample_geom",f"{type}"]]
        
                 
                        
            
def get_rgb_code(value,type="carbon"):
    if type=="carbon":
        if value<=250:
            return 127, 255, 212
        if value >250  and value<=500:
            return 64, 224, 208
        if value >500  and value<=1000:
            return 0, 255, 255
        if value>1000 and value<=1500:
            return 135, 206, 235
        if value>1500:
            return 0, 0, 128
    elif type=="biomass":
        if value<=0.5:
            return 255, 255, 102
        if value >0.5  and value<=1:
            return 255, 204, 0
        if value >1  and value<=1.5:
            return 154, 205, 50
        if value>1.5 and value<=2:
            return 173, 255, 47
        if value>2:
            return 34, 139, 34
    else:
        raise ValueError("type not valid")
        
def create_tiff(polygon,filename,zoom=21):
    bbox=list(polygon.bounds)
    tms_to_geotiff(output=filename, bbox=bbox, zoom=zoom, source='Satellite',overwrite=True)
    return filename
# Light Yellow: #FFFF66
# Bright Yellow: #FFCC00
# Yellow-Green: #9ACD32
# Lemon-Lime: #ADFF2F
# Forest Green: #228B22
# def get_rgb_code(carbon):
#     if carbon<=0.5:
#         return 255, 255, 102
#     if carbon >0.5  and carbon<=1:
#         return 255, 204, 0
#     if carbon >1  and carbon<=1.5:
#         return 154, 205, 50
#     if carbon>1.5 and carbon<=2:
#         return 173, 255, 47
#     if carbon>2:
#         return 34, 139, 34
@contextmanager
def resample_raster(raster, scale=6):
    t = raster.transform

    # rescale the metadata
    transform = Affine(t.a / scale, t.b, t.c, t.d, t.e / scale, t.f)
    height = raster.height * scale
    width = raster.width * scale

    profile = raster.profile
    profile.update(transform=transform, driver='GTiff', height=height, width=width)

    data = raster.read( # Note changed order of indexes, arrays are band, row, col order not row, col, band
            out_shape=(raster.count, height, width),
            #resampling
            resampling=Resampling.cubic_spline,
        )

    with MemoryFile() as memfile:
        with memfile.open(**profile) as dataset: # Open as DatasetWriter
            dataset.write(data)
            del data

        with memfile.open() as dataset:  # Reopen as DatasetReader
            yield dataset  # Note yield not return     
#add_carbon_to_file("../csv_data/touba-toul-trees-2023-03-25.csv")      
def create_vrt(coordinates,raster_data,tree_masks,name_vrt,ndvi_threshold=0.20,type="carbon",vrts_path="../physic_parameter"):
    
    shape=Polygon(coordinates)
    tree_polygon=[]
    carbon_tree=[]

    with MemoryFile(raster_data) as memfile:
        with memfile.open() as dst:
            out_image, out_transform = rasterio.mask.mask(dst, [shape, ], crop=True, nodata=-1)
            out_meta = dst.meta.copy()
            epsg_code = int(dst.crs.data['init'][5:])
            out_meta.update({"driver": "GTiff",
                             "height": out_image.shape[1],
                             "width": out_image.shape[2],
                             "transform": out_transform,
                             "crs": pycrs.parse.from_epsg_code(epsg_code).to_proj4()}
                            )
            with MemoryFile() as another_memfile:
                with another_memfile.open(**out_meta) as dataset:
                    dataset.write(out_image) 
                    transform = dataset.transform
                    width = dataset.width
                    height = dataset.height
                    rgba_composite=np.empty((height, width,4),dtype=np.uint8)
                    rgba_composite.fill(np.nan)
                    # Get the width and height of the raster
                    b,g,r,n=dataset.read([2,3,4,8],dtype=np.uint8)
                    rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
                    # Convert the row and column indices to x and y coordinates in the CRS of the raster dataset
                    x, y = rasterio.transform.xy(transform, rows.flatten(), cols.flatten())

                    lon, lat =x,y
                    lon = np.asarray(x).reshape((height, width))[0]
                    lat = np.asarray(y).reshape((height, width))[:,0]
                    nbr_tree=0
                    for P in tree_masks:
                        bbox=P.bounds
                        circle=polygon_to_circle(P)
                        if not shape.intersects(P):
                            pass
                        _,min_x_index=find_closest_number_left(bbox[0],lon)
                        #latitude value is in a descending order so the max have a smaller index so we have to permut min_
                        _,min_y_index=find_closest_number_left(bbox[1],lat)
                        _,max_x_index=find_closest_number_left(bbox[2],lon)
                        _,max_y_index=find_closest_number_left(bbox[3],lat)
                        _r=r[max_y_index:min_y_index,min_x_index:max_x_index]      
                        _n=n[max_y_index:min_y_index,min_x_index:max_x_index]   
                        ndvi=(_n-_r)/(_n+_r)
                        logger.error(f'ndvi mean= {ndvi.mean()}')
                        if len(ndvi)!=0:
                            logger.error(f'ndvi mean= {ndvi.mean()}')
                        if len(ndvi)!=0 and ndvi.mean()>ndvi_threshold:
                            logger.success(f'area= {P.area}')
                            logger.error(f'ndvi= {ndvi.mean()}')
                            carbon_value=compute_physic_parameter(P,type)
                            carbon_tree.append(carbon_value)
                            tree_polygon.append(P)
                            nbr_tree+=1
                            for i in range(max_y_index,min_y_index +1):
                                for j in range(min_x_index,max_x_index +1):
                                    if r[i,j]==-1:
                                        pass
                                    point=Point(lon[j],lat[i])
                                    if P.contains(point):
                                        r_i,g_i,b_i=get_rgb_code(carbon_value,type)
                                        rgba_composite[i,j,0]=r_i
                                        rgba_composite[i,j,1]=g_i
                                        rgba_composite[i,j,2]=b_i
                                        rgba_composite[i,j,3]=255
                    logger.success(f'nbr trees = {nbr_tree}')
                    img = Image.fromarray(rgba_composite)
                    os.makedirs(vrts_path,exist_ok=True)
                    filename=os.path.join(vrts_path,f'{name_vrt}.png')
                    img.save(filename)
            
            return pd.DataFrame({"sample_geom":tree_polygon,"carbon":carbon_tree})
         

def create_vrt_parallel(coordinates,raster_data,mask_generator,filenames,name_vrt,ndvi_threshold=0.20,type="carbon",vrts_path="../physic_parameter",min_tree_area=15,max_tree_area=600):
    
    shape=Polygon(coordinates)
    tree_polygons=[]
    #carbon_tree=[]

    with MemoryFile(raster_data) as memfile:
        with memfile.open() as dataset:
            # out_image, out_transform = mask(dst, [shape, ], crop=True, nodata=-1)
            # out_meta = dst.meta.copy()
            # epsg_code = int(dst.crs.data['init'][5:])
            # out_meta.update({"driver": "GTiff",
            #                  "height": out_image.shape[1],
            #                  "width": out_image.shape[2],
            #                  "transform": out_transform,
            #                  "crs": pycrs.parse.from_epsg_code(epsg_code).to_proj4()}
            #                 )
            # with MemoryFile() as another_memfile:
            #     with another_memfile.open(**out_meta) as dataset:
                        #datasfet.write(out_image) 
            transform = dataset.transform
            width = dataset.width
            height = dataset.height
            rgba_composite=np.zeros((height, width,4),dtype=np.uint8)
            
            #rgba_composite.fill(np.nan)
            # Get the width and height of the raster
            b,g,r,n=dataset.read([2,3,4,8])
            rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
            # Convert the row and column indices to x and y coordinates in the CRS of the raster dataset
            x, y = rasterio.transform.xy(transform, rows.flatten(), cols.flatten())
            lon, lat =x,y
            lon = np.asarray(x).reshape((height, width))[0]
            lat = np.asarray(y).reshape((height, width))[:,0]
            
            def get_only_trees_masks(filename):
                masks=create_tree_masks(mask_generator,filename,min_tree_area,max_tree_area)
                def get_tree(polygon):
                    bbox=polygon.bounds
                    circle=polygon_to_circle(polygon)
                    if not shape.intersects(polygon):
                        pass
                    _,min_x_index=find_closest_number_left(bbox[0],lon)
                    #latitude value is in a descending order so the max have a smaller index so we have to permut min_
                    _,min_y_index=find_closest_number_left(bbox[1],lat)
                    _,max_x_index=find_closest_number_left(bbox[2],lon)
                    _,max_y_index=find_closest_number_left(bbox[3],lat)
                    _r=r[max_y_index:min_y_index,min_x_index:max_x_index]      
                    _n=n[max_y_index:min_y_index,min_x_index:max_x_index]   
                    ndvi=(_n-_r)/(_n+_r)    
                    if len(ndvi)==0:
                        return None
                    if len(ndvi)!=0:
                        if ndvi.mean()<ndvi_threshold:                    
                            return None
                    carbon_value=compute_physic_parameter(polygon,type)
                    for i in range(max_y_index,min_y_index +1):
                        for j in range(min_x_index,max_x_index +1):
                            if r[i,j]==-1:
                                pass
                            point=Point(lon[j],lat[i])
                            if circle.contains(point):
                                r_i,g_i,b_i=get_rgb_code(carbon_value,type)
                                rgba_composite[i,j,0]=r_i
                                rgba_composite[i,j,1]=g_i
                                rgba_composite[i,j,2]=b_i 
                                rgba_composite[i,j,3]=255 
                    return polygon
                trees=Parallel(n_jobs=-1)(delayed(get_tree)(polygon)  for polygon in masks)    
             
                return list(filter(lambda item: item is not  None ,trees))
            
            trees=Parallel(n_jobs=-1,backend="threading")(delayed(get_only_trees_masks)(filename)  for filename in filenames)
            tree_polygons.expend(trees)
            carbon_values=Parallel(n_jobs=-1)(delayed(compute_physic_parameter)(polygon)  for polygon in tree_polygons)
            img = Image.fromarray(rgba_composite)
            os.makedirs(vrts_path,exist_ok=True)
            filename=os.path.join(vrts_path,f'{name_vrt}.png')
            img.save(filename)
            return pd.DataFrame({"sample_geom":tree_polygons,"carbon":carbon_values})

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

def get_box_4326(filename,rgb_path="RGBdata"):
    gdf = gpd.read_file(filename)
    #new_name=os.path.join(output_path,filename.split("/")[-1].split(".")[0])+".geojson"
    tiff_rgb_filename=os.path.join(rgb_path,filename.split("/")[-1].split(".")[0])+".tiff"
    gdf["geometry"] = gdf.buffer(0)

    # Reproject to Web Mercator (EPSG:4326)
    gdf_web_mercator = gdf.to_crs(epsg=4326)
    polygon_coords = list(gdf_web_mercator.geometry.iloc[0].exterior.coords)
    create_tiff(Polygon(polygon_coords),tiff_rgb_filename)
    # Create a geodataframe and save as GeoJSON
    #gdf = gpd.GeoDataFrame({'geometry': [polygon]}, crs=crs)
    #geojson_path = os.path.splitext(local_path)[0] + ".geojson"
    #gdf_web_mercator.to_file(new_name, driver="GeoJSON")
            
def get_all_bbox(geojson_path="../geojson_path/chm4326",rgb_path="../RGBdata"):
    filenames=os.listdir(geojson_path)
    print("yess")
   # Parallel(n_jobs=-1)(delayed(get_box_4326)(os.path.join(geojson_path,filename),rgb_path) for filename in filenames)
    for filename in filenames:
        get_box_4326(os.path.join(geojson_path,filename),rgb_path)           
#def download_tiff(geojson_output):
    
get_all_bbox()

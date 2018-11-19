#!/home/tq/anaconda3/bin/python
# -*- coding:utf-8 -*-
import os
import sys
import math
import glob
import logging
import common
import numpy as np
from osgeo import gdal, osr, gdalnumeric, ogr
import subprocess

from traindata_extractor.general.common import *
from traindata_extractor.general.calc_mask_by_shape import *


def stat_area_zonal(raster_path: str, shp_path: str):
    """
    """
    # open vector shape_path
    try:
        shapef = ogr.Open(shp_path)
        lyr = shapef.GetLayer(0)
        poly = lyr.GetNextFeature()
        lyrdn = lyr.GetLayerDefn()
        maxpoly = lyr.GetFeatureCount()
    except Exception as e:
        raise Exception("open shape failed!")

    # open raster
    ds = gdal.Open(raster_path)
    geo_trans = ds.GetGeoTransform()
    x_size = ds.RasterXSize
    y_size = ds.RasterYSize
    img_shape = [x_size, y_size]
    data = ds.GetRasterBand(1).ReadAsArray()

    lyr.ResetReading()
    npoly = 0
    tmplist = []
    for poly in lyr:
        # loop poly in lyr, draw ROIs in Image
        polyname = poly.GetField("NAME_2")
        npoly = npoly + 1

        mask, num_label, list_label = calc_mask_by_shape(
            shp_path,
            geo_trans,
            img_shape,
            specified_field="NAME_2",
            condition=[polyname],
            mask_value=1,
            field_strict=True,
        )
        if mask is None:
            raise Exception("mask is wrong")

        pixels = get_valid_data(data, mask)
        idx = np.where(pixels == 1)
        n_vali_pixels = len(idx[0])
        tmplist.append([polyname, n_vali_pixels])

    for l in tmplist:
        print("{},\t{}".format(l[0], l[1]))
    return True


def stat_area_zonal_county(raster_path: str, shp_path: str):
    """
    """
    # open vector shape_path
    try:
        shapef = ogr.Open(shp_path)
        lyr = shapef.GetLayer(0)
        poly = lyr.GetNextFeature()
        lyrdn = lyr.GetLayerDefn()
        maxpoly = lyr.GetFeatureCount()
    except Exception as e:
        raise Exception("open shape failed!")

    # open raster
    ds = gdal.Open(raster_path)
    geo_trans = ds.GetGeoTransform()
    x_size = ds.RasterXSize
    y_size = ds.RasterYSize
    img_shape = [x_size, y_size]
    data = ds.GetRasterBand(1).ReadAsArray()

    lyr.ResetReading()
    npoly = 0
    tmplist = []
    for poly in lyr:
        # loop poly in lyr, draw ROIs in Image
        polyname = poly.GetField("NAME_3")
        polyname2 = poly.GetField("NL_NAME_3")
        npoly = npoly + 1

        mask, num_label, list_label = calc_mask_by_shape(
            shp_path,
            geo_trans,
            img_shape,
            specified_field="NAME_3",
            condition=[polyname],
            mask_value=1,
            field_strict=True,
        )
        if mask is None:
            raise Exception("mask is wrong")

        pixels = get_valid_data(data, mask)
        idx = np.where(pixels == 1)
        n_vali_pixels = len(idx[0])
        tmplist.append([polyname, polyname2, n_vali_pixels])

    for l in tmplist:
        print("{},{},\t{}".format(l[0], l[1], l[2]))
    return True


def stat_area_zonal_county15(raster_path: str, shp_path: str):
    """
    use 2015 county bound shp
    """
    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "NO")
    gdal.SetConfigOption("SHAPE_ENCODING", "")  # chinese char
    ogr.RegisterAll()

    # open vector shape_path
    try:
        driver = ogr.GetDriverByName("ESRI Shapefile")
        shapef = driver.Open(shp_path)
        # shapef = ogr.Open(shp_path)
        lyr = shapef.GetLayer(0)
        poly = lyr.GetNextFeature()
        lyrdn = lyr.GetLayerDefn()
        maxpoly = lyr.GetFeatureCount()
    except Exception as e:
        raise Exception("open shape failed!")

    # open raster

    ds = gdal.Open(raster_path)
    geo_trans = ds.GetGeoTransform()
    x_size = ds.RasterXSize
    y_size = ds.RasterYSize
    img_shape = [x_size, y_size]
    data = ds.GetRasterBand(1).ReadAsArray()

    lyr.ResetReading()
    npoly = 0
    tmplist = []
    for poly in lyr:
        # loop poly in lyr, draw ROIs in Image
        polyname = poly.GetField("label")
        npoly = npoly + 1

        mask, num_label, list_label = calc_mask_by_shape(
            shp_path,
            geo_trans,
            img_shape,
            specified_field="label",
            condition=[polyname],
            mask_value=1,
            field_strict=True,
        )
        if mask is None:
            raise Exception("mask is wrong")

        pixels = get_valid_data(data, mask)
        idx = np.where(pixels == 1)
        n_vali_pixels = len(idx[0])
        tmplist.append([polyname, n_vali_pixels])

    for l in tmplist:
        print("{},\t{}".format(l[0], l[1]))
    return True


def get_valid_data(
    data: np.ndarray,
    mask: np.ndarray,
    *,
    nodata_value: list = [0],
    validRange: list = [0, 10000],
):
    """

    """
    mask_idx = np.where(mask > 0)
    train_data = data[mask_idx]
    mask_t = mask[mask_idx]
    if train_data.shape == mask_t.shape:
        pass  # print(train_data.shape)
    else:
        raise Exception("get_valid_data_new(): shape not match! skip")

    return train_data.flatten()

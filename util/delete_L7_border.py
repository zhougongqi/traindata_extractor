import os
import gc
import cv2
import sys
import math
import time
import glob
import logging
import common
import numpy as np
from osgeo import gdal, osr, gdalnumeric
import subprocess

from traindata_extractor.general.common import *

home_dir = os.path.expanduser("~")


def delete_L7_border(process_dict: dict):
    """
    stack waterfall results (WF) and landsat 8 original image,
    delete l-7 ugly border using l-8 mask
    """
    # get file list
    mlp_list = process_dict["wf_list"]
    qa_path_list = load_json(process_dict["l8path"])
    qa_dict = {}
    for q in qa_path_list:
        qa_dict[q[0]] = home_dir + "/" + q[1]

    tt_mlp = len(mlp_list)
    n_mlp = 0
    for mlp in mlp_list:
        n_mlp += 1
        print("{}/{} mlp result".format(n_mlp, tt_mlp))

        # get path row
        mlp_bname = os.path.basename(mlp)
        mlp_name, mlp_ext = os.path.splitext(mlp_bname)
        nameblock = mlp_name.split("-")
        path = nameblock[0]  # path number
        row2 = nameblock[1]  # row number
        pathrowstr = path + "/" + row2

        # find wanted l8 tile
        print(pathrowstr)
        qa_file = qa_dict[pathrowstr]
        if not qa_file.endswith("_pixel_qa.tif"):
            raise Exception("qa file error")

        if qa_file is None:
            print("file not found, skip~ ")
            continue

        # run gdal_merge.py to stack them
        tmp_file = process_dict["work_path"] + "tmp/" + mlp_bname + "_tmp.tif"
        cmd_str = (
            "gdal_merge.py -separate -of GTiff -o "  # -n 0 -a_nodata 0
            + tmp_file
            + " "
            + qa_file
            + " "
            + mlp
        )
        print("cmd string is :", cmd_str)
        process_status = subprocess.run(cmd_str, shell=True)
        if process_status.returncode != 0:
            print("cmd failed!")
            return None

        # read from new generated tmp.tif
        ds = gdal.Open(tmp_file)
        geo_trans = ds.GetGeoTransform()
        w = ds.RasterXSize
        h = ds.RasterYSize
        img_shape = [h, w]
        proj = ds.GetProjection()
        ras_spatial_ref = osr.SpatialReference(wkt=proj)
        proj_name = ras_spatial_ref.GetAttrValue("projcs")
        qa_data = ds.GetRasterBand(1).ReadAsArray()
        mlp_data = ds.GetRasterBand(2).ReadAsArray()

        # print("mlp: \t", img_shape, proj_name)
        qa_mask = qa_data.copy()
        qa_mask[qa_mask < 0] = 0
        qa_mask[qa_mask > 0] = 1

        mlp_data = mlp_data * qa_mask

        # build output path
        outpath = process_dict["work_path"] + mlp_name + "_no_L7_border.tif"
        # write output into tiff file
        out_ds = gdal.GetDriverByName("GTiff").Create(outpath, w, h, 1, gdal.GDT_Int16)
        out_ds.SetProjection(proj)
        out_ds.SetGeoTransform(geo_trans)
        out_ds.GetRasterBand(1).WriteArray(mlp_data)
        out_ds.FlushCache()
        ds = None
    print("fin")


def delete_L7_border_further(process_dict: dict):
    """
    stack waterfall results (WF) and landsat 8 original image,
    delete l-7 ugly border using l-8 mask

    and further remove borders (meant shrink l-8 coverage by about 115 horizontally)
    """
    # get file list
    mlp_list = process_dict["wf_list"]
    qa_path_list = load_json(process_dict["l8path"])
    qa_dict = {}
    for q in qa_path_list:
        qa_dict[q[0]] = home_dir + "/" + q[1]

    tt_mlp = len(mlp_list)
    n_mlp = 0
    for mlp in mlp_list:
        n_mlp += 1
        print("{}/{} mlp result".format(n_mlp, tt_mlp))

        # get path row
        mlp_bname = os.path.basename(mlp)
        mlp_name, mlp_ext = os.path.splitext(mlp_bname)
        nameblock = mlp_name.split("-")
        path = nameblock[0]  # path number
        row2 = nameblock[1]  # row number
        pathrowstr = path + "/" + row2

        # find wanted l8 tile
        print(pathrowstr)
        qa_file = qa_dict[pathrowstr]

        if qa_file is None:
            print("file not found, skip~ ")
            continue

        # run gdal_merge.py to stack them
        tmp_file = process_dict["work_path"] + "tmp/" + mlp_bname + "_tmp.tif"
        cmd_str = (
            "gdal_merge.py -separate -of GTiff -o "  # -n 0 -a_nodata 0
            + tmp_file
            + " "
            + qa_file
            + " "
            + mlp
        )
        print("cmd string is :", cmd_str)
        process_status = subprocess.run(cmd_str, shell=True)
        if process_status.returncode != 0:
            print("cmd failed!")
            return None

        # read from new generated tmp.tif
        ds = gdal.Open(tmp_file)
        geo_trans = ds.GetGeoTransform()
        w = ds.RasterXSize
        h = ds.RasterYSize
        img_shape = [h, w]
        proj = ds.GetProjection()
        ras_spatial_ref = osr.SpatialReference(wkt=proj)
        proj_name = ras_spatial_ref.GetAttrValue("projcs")
        qa_data = ds.GetRasterBand(1).ReadAsArray()
        mlp_data = ds.GetRasterBand(2).ReadAsArray()

        # print("mlp: \t", img_shape, proj_name)
        print(qa_data.shape)
        qa_mask = qa_data.copy()
        qa_mask[qa_mask < 0] = 0
        qa_mask[qa_mask > 0] = 1

        for ih in range(h):
            qatmp = qa_mask[ih, :]
            n1 = np.sum(qatmp)
            if n1 <= 230:
                qatmp[:] = 0
            else:  # shrink
                idx1 = np.where(qatmp == 1)
                qatmp[idx1[0][0] : idx1[0][0] + 115] = 0
                qatmp[idx1[0][-1] - 114 : idx1[0][-1] + 1] = 0
            print_progress_bar(ih + 1, h)
        print(" ")

        mlp_data = mlp_data * qa_mask

        # build output path
        outpath = process_dict["work_path"] + mlp_name + "_absolutely_no_L7_border.tif"
        # write output into tiff file
        out_ds = gdal.GetDriverByName("GTiff").Create(outpath, w, h, 1, gdal.GDT_Int16)
        out_ds.SetProjection(proj)
        out_ds.SetGeoTransform(geo_trans)
        out_ds.GetRasterBand(1).WriteArray(mlp_data)
        out_ds.FlushCache()
        ds = None
    print("fin")


if __name__ == "__main__":
    wf_dir = "/home/tq/data_pool/cleanup/waterfall_data/crop_tif/Landsat/20180401-20180930/china-p10-2018/"
    wf_list = glob.glob(wf_dir + "*.tif")
    process_dict = {}
    process_dict["wf_list"] = wf_list
    process_dict[
        "l8path"
    ] = "/home/tq/data_pool/Y_ALL/crop_models/region_cover_tiles/LC08_tile_match.json"
    process_dict[
        "work_path"
    ] = "/home/tq/data_pool/cleanup/waterfall_data/crop_tif/Landsat/20180401-20180930/china-p10-2018/rocks/"

    delete_L7_border_further(process_dict)
    print("fin")

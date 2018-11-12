import os
import gc
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


def combine_with_qiangtu(combine_dict: dict):
    """
    find cloud mask in "qa_path",
    replace cloud pixel with qiangtu.
    """
    # get file list
    mlp_list = get_bands_into_a_list(
        combine_dict["results_path"], "MLPClassifier_*.tif"
    )
    qtu_list = get_bands_into_a_list(combine_dict["qiangtu_path"], "*.tif")
    qa_path_list = load_json(combine_dict["qa_path"])
    qa_list = []
    for q in qa_path_list:
        q = home_dir + "/" + q + "/"
        qaf = glob.glob(q + "*_pixel_qa.tif")[0]
        qa_list.append(qaf)

    tt_mlp = len(mlp_list)
    n_mlp = 0
    for mlp in mlp_list:
        n_mlp += 1
        print("{}/{} mlp result".format(n_mlp, tt_mlp))

        # get path row
        mlp_bname = os.path.basename(mlp)
        mlp_name, mlp_ext = os.path.splitext(mlp_bname)
        nameblock = mlp_name.split("_")
        pathrow = nameblock[4]
        path = pathrow[0:3]  # path number
        row = pathrow[3:6]  # row number
        row2 = row[1:3]  # row number 2 digits

        # find wanted qiangtu tile and qa tile
        qa_file = find_qa_by_pathrow(qa_list, path, row)
        qtu_file = find_file_by_pathrow(qtu_list, path, row2)
        print(qa_file)
        print(qtu_file)
        print("-" * 50)

        if qa_file is None or qtu_file is None:
            print("file not found, skip~ ")
            continue

        # read all files
        # # mlp results
        # ds = gdal.Open(mlp)
        # geo_trans = ds.GetGeoTransform()
        # w = ds.RasterXSize
        # h = ds.RasterYSize
        # img_shape = [h, w]
        # proj = ds.GetProjection()
        # ras_spatial_ref = osr.SpatialReference(wkt=proj)
        # proj_name = ras_spatial_ref.GetAttrValue("projcs")
        # mlp_data = ds.ReadAsArray()
        # ds = None
        # print("mlp: \t", img_shape, proj_name)

        # # qiangtu
        # ds = gdal.Open(qtu_file)
        # geo_trans = ds.GetGeoTransform()
        # w = ds.RasterXSize
        # h = ds.RasterYSize
        # img_shape = [h, w]
        # proj = ds.GetProjection()
        # ras_spatial_ref = osr.SpatialReference(wkt=proj)
        # proj_name = ras_spatial_ref.GetAttrValue("projcs")
        # qtu_data = ds.ReadAsArray()
        # ds = None
        # print("qiangtu: \t", img_shape, proj_name)

        # # qa
        # ds = gdal.Open(qa_file)
        # geo_trans = ds.GetGeoTransform()
        # w = ds.RasterXSize
        # h = ds.RasterYSize
        # img_shape = [h, w]
        # proj = ds.GetProjection()
        # ras_spatial_ref = osr.SpatialReference(wkt=proj)
        # proj_name = ras_spatial_ref.GetAttrValue("projcs")
        # qa_data = ds.ReadAsArray()
        # ds = None
        # print("qa: \t", img_shape, proj_name)

        # run gdal_merge.py to stack them
        tmp_file = combine_dict["tmp_path"] + mlp_bname + "_tmp.tif"
        cmd_str = (
            "gdal_merge.py -separate -of GTiff -o "  # -n 0 -a_nodata 0
            + tmp_file
            + " "
            + qa_file
            + " "
            + mlp
            + " "
            + qtu_file
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
        qtu_data = ds.GetRasterBand(3).ReadAsArray()

        # print("mlp: \t", img_shape, proj_name)
        qa_mask = qa_data.copy()
        qa_mask[qa_mask == 1] = 0
        qa_mask[qa_mask > 1] = 1

        qa_data[qa_data > 326] = 999
        qa_data[qa_data == 1] = 999
        qa_data[qa_data < 326] = 1  # cloud is 0
        qa_data[qa_data == 999] = 0
        print(np.min(qa_data), np.max(qa_data))

        qtu_data[qtu_data == 255] = 0

        # # read mlp proj and geotrans
        # ds = gdal.Open(mlp)
        # geo_trans = ds.GetGeoTransform()
        # w = ds.RasterXSize
        # h = ds.RasterYSize
        # img_shape = [h, w]
        # proj = ds.GetProjection()
        # ds = None

        # fusion
        out = gdalnumeric.choose(qa_data, (qtu_data, mlp_data))
        out = out * qa_mask

        out = out + 1
        idx = np.where(qa_data == 0)
        out[idx] = 0
        out = out * qa_mask
        # build output path
        outpath = combine_dict["work_path"] + mlp_name + "_fusion.tif"
        # write output into tiff file
        out_ds = gdal.GetDriverByName("GTiff").Create(outpath, w, h, 1, gdal.GDT_Byte)
        out_ds.SetProjection(proj)
        out_ds.SetGeoTransform(geo_trans)
        out_ds.GetRasterBand(1).WriteArray(out)
        out_ds.FlushCache()
        ds = None
    print("fin")

    #

    # loop each file
    return True


def find_file_by_pathrow(inputlist: list, path: str, row: str):
    sstr = path + "-" + row + "-"
    outfile = None
    for f in inputlist:
        if f.find(sstr) >= 0:
            outfile = f
    if outfile is None:
        print("file not found")
        return None
    return outfile


def find_qa_by_pathrow(inputlist: list, path: str, row: str):
    sstr = path + row
    outfile = None
    for f in inputlist:
        if f.find(sstr) >= 0:
            outfile = f
    if outfile is None:
        print("file not found")
        return None
    return outfile


def combine_with_qiangtu_step2(z_path: str, q_path: str, work_path: str):
    """
    z_path: path of my rice, and be merged by arcgis
        0: to be replaced by qiangtu
        1: not rice
        2: rice
    q_path: qiangtu path, merged
        0: not rice
        1: rice
        255: not rice
    """
    # read all files
    # run gdal_merge.py to stack them
    mlp_bname = os.path.basename(z_path)
    mlp_name, mlp_ext = os.path.splitext(mlp_bname)

    tmp_file = work_path + mlp_name + "_tmp.tif"
    cmd_str = (
        "gdal_merge.py -separate -of GTiff -o "  # -n 0 -a_nodata 0
        + tmp_file
        + " "
        + z_path
        + " "
        + q_path
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
    mlp_data = ds.GetRasterBand(1).ReadAsArray()
    qtu_data = ds.GetRasterBand(2).ReadAsArray()

    qtu_data[qtu_data == 255] = 0
    qtu_data = qtu_data + 1

    idx = np.where(mlp_data == 0)
    mlp_data[idx] = qtu_data[idx]

    mlp_data = mlp_data - 1
    mlp_data[mlp_data > 200] = 0
    # build output path
    outpath = work_path + mlp_name + "_fusion_final.tif"
    # write output into tiff file
    out_ds = gdal.GetDriverByName("GTiff").Create(outpath, w, h, 1, gdal.GDT_Byte)
    out_ds.SetProjection(proj)
    out_ds.SetGeoTransform(geo_trans)
    out_ds.GetRasterBand(1).WriteArray(mlp_data)
    out_ds.FlushCache()
    ds = None

    return True


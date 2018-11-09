import os, sys
import glob
import pandas as pd
import common
from osgeo import gdal
from traindata_extractor.general.common import *

home_dir = os.path.expanduser("~")

if __name__ == "__main__":
    """
    prepare some annoying steps

    1, read Landsat-8 data list from a json
    2, reform the path make them looks alright
    3, get bands from each l-8 paths, and stack them
    """
    # excel_path = "/home/tq/data_pool/china_crop/Liaoning/liaoning_data_path.xls"
    # "/home/tq/data_pool/china_crop/Fuyu/fuyu_data.xls"
    work_path = "/home/tq/data_pool/china_crop/Jilin-Heilongjiang/L8/"

    # 1 read excel
    # sheet1 = pd.read_excel(excel_path, sheetname=0)
    # df = pd.DataFrame(sheet1)
    # filelist = df[["SR_data Path"]].values.T.tolist()[:][0]  # 数据来源
    filelist = load_json("/home/tq/data_pool/X-EX/china/JL/JL_result_2018.json")
    print(filelist)

    # 2
    tmlist = []
    for f in filelist:
        if f.find("LC08") != -1:
            tmp = f.replace('"', "")
            tmp = tmp.replace(",", "")
            tmp = tmp.replace(" ", "")
            if tmp.endswith("/"):
                pass
            else:
                tmp = tmp + "/"
            tmlist.append(os.path.join(home_dir, tmp))

    print(tmlist)

    # 3

    for tm in tmlist:
        tm_shortname = tm.split("/")[-2]
        print(tm)
        band_list = get_bands_into_a_list(tm, "*sr_band*.tif")
        nbands = len(band_list)
        bn = 0

        # read the first one to get some paras
        ds = gdal.Open(band_list[0])
        geo_trans = ds.GetGeoTransform()
        w = ds.RasterXSize
        h = ds.RasterYSize
        img_shape = [h, w]
        proj = ds.GetProjection()
        data = ds.ReadAsArray()
        ds = None
        if "int8" in data.dtype.name:
            datatype = gdal.GDT_Byte
        elif "int16" in data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        # build output path
        outpath = work_path + tm_shortname + "_stacked.tif"
        # write output into tiff file
        out_ds = gdal.GetDriverByName("GTiff").Create(outpath, w, h, nbands, datatype)

        for band in band_list:
            bn += 1
            print("  {}/{} bands".format(bn, nbands))
            # read files and stack them
            ds = gdal.Open(band)
            geo_trans = ds.GetGeoTransform()
            w = ds.RasterXSize
            h = ds.RasterYSize
            img_shape = [h, w]
            data = ds.ReadAsArray()
            # write
            out_ds.GetRasterBand(bn).WriteArray(data)
            ds = None
        out_ds.SetProjection(proj)
        out_ds.SetGeoTransform(geo_trans)
        out_ds.FlushCache()
        out_ds = None

    print("fin")

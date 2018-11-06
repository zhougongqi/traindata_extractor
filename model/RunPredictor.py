import os
import gc
import sys
import math
import time
import logging
import numpy as np
from osgeo import gdal
from sklearn.externals import joblib

from traindata_extractor.general.feat_calc import feat_calc
from traindata_extractor.general.common import *

my_logger = logging.getLogger(__name__)


def run_predictor(
    process_dict, *, start: int = 0, end: int = -1, coef: float = 10000.0
) -> (bool, str):
    """
    Function:
        Run saved model to produce classification results
    Input:
        process_dict, where useful paras are below:
            model_path: path stores the .m file
            pro_ras_list: list of raster dir where contains the raster bands to be 
                        processed (for Landsat 8 data stored in seperate bands)
            work_path: stores the tmp files and final results.
            read_order_list: list of read order

        * optional parameters:
        start: start line
        end: end line
    Output:
        True for success!
        result stored in $work_path
    """
    # set para
    model_path = process_dict["model_path"]
    pro_ras_list = process_dict["pro_ras_list"]
    work_path = process_dict["work_path"]
    read_order_list = process_dict["read_order_list"]

    single_flag = False
    model_shortname = os.path.basename(model_path)
    model_label = model_shortname.split("_")[0]

    # load model
    predictor = joblib.load(model_path)

    # get traindata_key_dic from read_order_list
    traindata_key_dic = get_traindata_key_dict(read_order_list)

    # begin to predict

    # loop each file in list
    n_files = len(pro_ras_list)
    nf = 0
    for ras_file in pro_ras_list:
        nf += 1
        print("processing img {}/{} :{}".format(nf, n_files, ras_file))
        img_label = ras_file.split("/")[-2]
        # get band tiffs into a list
        bands_dict, n_feat = get_bands_into_a_dict(ras_file)

        # read one input raster dataset
        ds = gdal.Open(next(bands_dict.walk())[-1])
        geo_trans = ds.GetGeoTransform()
        w = ds.RasterXSize
        h = ds.RasterYSize
        img_shape = [h, w]
        # my_logger.info("img shape: %s", img_shape)

        # prepare data
        data = prepare_data(bands_dict, read_order_list, n_feat)
        if data is None:
            my_logger.error("error in preparing data!")
            return False, None

        # loop to predict
        nlines = data.shape[0]
        out_arr = np.zeros(img_shape).flatten()
        for dh in range(h):
            t = predictor.predict(data[dh * w : dh * w + w, :] / coef)
            out_arr[dh * w : dh * w + w] = t
            print_progress_bar(dh + 1, h)
        out_arr = out_arr.reshape(-1, w)
        print("done!")

        # build output path
        outpath = (
            work_path
            + model_label
            + "_result_"
            + img_label
            + "_P"
            + time.strftime("%Y%m%d%H%M%S", time.localtime())
            + ".tif"
        )

        out_arr = out_arr.astype(np.int8)
        # write output into tiff file
        out_ds = gdal.GetDriverByName("GTiff").Create(outpath, w, h, 1, gdal.GDT_Byte)
        out_ds.SetProjection(ds.GetProjection())
        out_ds.SetGeoTransform(geo_trans)
        out_ds.GetRasterBand(1).WriteArray(out_arr)
        out_ds.FlushCache()
        my_logger.info("file write finished!")

    return True, outpath


def prepare_data(raster_data_dic: dict, read_order_list: list, n_feat: int):
    """
    Function:
        get ndarray from raster images in $raster_data_dic
    Input:
        raster_data_dic:  a 2-layered dict contains the raster files
                       as the model input.
                       like this:
                        {"sensor-1":{"band-1":"band path",
                                     "band-2":"band path",
                                    ...}
                         "sensor-2":{"band-1":"band path",
                                     "band-2":"band path",
                                    ...}
                        }
        read_order_list: list of read order
    Output:
        an ndarray contains all selected raster lines in shape of (:,1)
    """
    # open first raster file to get some infos
    try:
        first_raster_name = next(raster_data_dic.walk())[-1]
        ds = gdal.Open(first_raster_name)
        w = ds.RasterXSize
        h = ds.RasterYSize
        img_shape = [h, w]
    except Exception as e:
        my_logger.error("open file error: %s", first_raster_name)
        return None

    # get array from files in dictionary
    array = np.zeros([w * h, n_feat])
    icol = 0
    keys = []
    for ro in read_order_list:
        k1 = ro[0]
        k2 = ro[1]
        # if k1 not in raster_data_dic.keys():
        #     my_logger.error("key error: {}".format(k1))
        #     return None
        for k in raster_data_dic.keys():
            keys.append(k)
        k1 = keys[0]
        try:
            ds = gdal.Open(raster_data_dic[k1][k2])
            data = ds.ReadAsArray().reshape(-1, 1)
        except Exception as e:
            my_logger.error("open file error: {}".format(raster_data_dic[k1][k2]))
            return None
        array[:, icol] = data.flatten()
        icol += 1
        print(icol)

    array = replace_invalid_value(array, 0)
    return array


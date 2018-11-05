import os
import gc
import sys
import math
import time
import logging
import numpy as np
from osgeo import gdal
from sklearn.externals import joblib
import logging
from vortex.general.feat_calc import feat_calc
from vortex.general.common import get_traindata_key_dict
from vortex.general.common import replace_invalid_value


def run_predictor(process_dict, *, start: int = 0, end: int = -1) -> (bool, str):
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

    # read raster dataset
    ds = gdal.Open(next(raster_data_dic.walk())[-1])
    geo_trans = ds.GetGeoTransform()
    w = ds.RasterXSize
    h = ds.RasterYSize
    img_shape = [h, w]
    my_logger.info("img shape: %s", img_shape)

    # calc number of feature
    n_feat = len(read_order_list)

    # begin to predict

    # loop each file in list
    n_files = len(pro_ras_list)
    nf = 0
    for ras_file in pro_ras_list:
        nf += 1

    my_logger.info("Begin to predict (no blocking) ...")
    data = prepare_data(
        raster_data_dic,
        bandmath_list,
        traindata_key_dic,
        read_order_list,
        n_feat,
        start,
        end,
    )
    if data is None:
        my_logger.error("error in preparing block data!")
        return False, None
    t = predictor.predict(data)
    t = t.reshape(-1, w)
    out_arr = t
    sys.stdout.write("done!\n")

    # build output path
    work_path = work_path  # + "model_result/"
    if not work_path:
        os.makedirs(work_path)
    outpath = (
        work_path
        + model_label
        + "_result_"
        + time.strftime("%Y%m%d%H%M%S", time.localtime())
        + ".tif"
    )
    if not work_path:
        my_logger.error(
            "RunPredictor_Block(): cannot find work path, please check again!"
        )
        return False, None
    out_arr = out_arr.astype(np.int8)

    # write output into tiff file
    out_ds = gdal.GetDriverByName("GTiff").Create(outpath, w, h, 1, gdal.GDT_Byte)
    out_ds.SetProjection(ds.GetProjection())
    out_ds.SetGeoTransform(geo_trans)
    out_ds.GetRasterBand(1).WriteArray(out_arr)
    out_ds.FlushCache()
    my_logger.info("file write finished!")

    return True, outpath

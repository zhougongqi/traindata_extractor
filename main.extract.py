import os
import glob
import pprint
import settings
from common import logger
from osgeo import gdal
from osgeo import osr, ogr

from traindata_extractor.ground_truth.TrainDataExtractor import TrainDataExtractorV2
from traindata_extractor.model.RunPredictor import run_predictor

from traindata_extractor.general.common import *
from traindata_extractor.general.Vividict import Vividict
from traindata_extractor.model.SvmModel import SVMClassifier
from traindata_extractor.model.skModel import skModel
from traindata_extractor.model.DecisionTree import DecisionTree

# test model
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier


def main(process_dict: dict):
    """
    this is a program process only traindata extraction
    """
    pprint.pprint(process_dict)
    img_prepro_dict = process_dict["img_pro_dict"]

    # get a read order list
    read_order_list, feat_list_1d = set_vvdic_key_order(img_prepro_dict)
    process_dict["read_order_list"] = read_order_list
    pprint.pprint(read_order_list)

    #
    tde = TrainDataExtractorV2(
        process_dict, read_order_list, sample_label="l8_heishan_test", is_binarize=False
    )
    status = tde.set_join_char("_")
    status = tde.set_keep_label([1, 2, 3, 4])
    traindata, feat_name_list, npypath = tde.go_get_mask_2npy()
    process_dict["traindata_path_npy"] = npypath

    # run predictor

    print(traindata[100:105, :])

    return True


def go_main(ori_ras_path: str):

    # "/home/tq/tq-data05/landsat_sr/LC08/01/118/032/LC08_L1TP_118032_20180827_20180830_01_T1/"
    # "/home/tq/tq-data03/landsat_sr/LC08/01/120/032/LC08_L1TP_120032_20180825_20180829_01_T1/"
    # "/home/tq/tq-data03/landsat_sr/LC08/01/119/030/LC08_L1TP_119030_20180802_20180814_01_T1/"
    ############
    ############ "/home/tq/tq-data04/landsat_sr/LC08/01/120/031/LC08_L1TP_120031_20180809_20180815_01_T1/"
    # "/home/tq/tq-data03/landsat_sr/LC08/01/118/029/LC08_L1TP_118029_20180811_20180815_01_T1/"
    # "/home/tq/tq-data05/landsat_sr/LC08/01/114/028/LC08_L1TP_114028_20180831_20180912_01_T1/"
    # "/home/tq/tq-data05/landsat_sr/LC08/01/117/027/LC08_L1TP_117027_20180820_20180829_01_T1/"
    # "/home/tq/tq-data05/landsat_sr/LC08/01/119/027/LC08_L1TP_119027_20180818_20180829_01_T1/"
    # "/home/tq/tq-data05/landsat_sr/LC08/01/115/028/LC08_L1TP_115028_20180806_20180814_01_T1/"
    outname_label = get_pathrow_data_label(ori_ras_path)
    # "/home/tq/tq-data04/landsat_sr/LC08/01/120/031/LC08_L1TP_120031_20180809_20180815_01_T1/"
    pro_ras_list = [
        # "/home/tq/tq-data04/landsat_sr/LC08/01/120/030/LC08_L1TP_120030_20180910_20180913_01_T1/"
        "/home/tq/tq-data04/landsat_sr/LC08/01/120/031/LC08_L1TP_120031_20180809_20180815_01_T1/"
    ]
    process_dict = {
        "outname_label": outname_label + "v5",
        "img_pro_dict": {},
        "shp_reproj_dict": {
            "samples": "/home/tq/data_pool/china_crop/Jilin-Heilongjiang/rois/roi_4c_RiOtSoCo_L8_utm_n51_HJv2.shp",
            "template": "/home/tq/data_pool/china_crop/Jilin-Heilongjiang/rois/roi_4c_RiOtSoCo_L8_utm_xxx_HJv2.shp",
        },
        "pro_ras_list": pro_ras_list,
        "work_path": "/home/tq/data_pool/china_crop/Jilin-Heilongjiang/npys/",
        "field_name": "label",
    }

    # glob wanted files
    img_pro_dict, band_num = get_bands_into_a_dict(ori_ras_path, "*sr_band*.tif")
    img_pro_list = get_bands_into_a_list(ori_ras_path, "*sr_band*.tif")
    process_dict["img_pro_dict"] = img_pro_dict

    # check proj
    ds = gdal.Open(img_pro_list[0])
    ras_proj = ds.GetProjection()
    ras_spatial_ref = osr.SpatialReference(wkt=ras_proj)
    proj_name = ras_spatial_ref.GetAttrValue("projcs")

    if proj_name.find("51N") > 0:
        shpname = process_dict["shp_reproj_dict"]["template"].replace("xxx", "n51")
        process_dict["shp_reproj_dict"]["samples"] = shpname
    elif proj_name.find("50N") > 0:
        shpname = process_dict["shp_reproj_dict"]["template"].replace("xxx", "n50")
        process_dict["shp_reproj_dict"]["samples"] = shpname
    elif proj_name.find("52N") > 0:
        shpname = process_dict["shp_reproj_dict"]["template"].replace("xxx", "n52")
        process_dict["shp_reproj_dict"]["samples"] = shpname
    else:
        shpname = process_dict["shp_reproj_dict"]["template"].replace("xxx", "n53")
        process_dict["shp_reproj_dict"]["samples"] = shpname

    shapef = ogr.Open(process_dict["shp_reproj_dict"]["samples"])
    lyr = shapef.GetLayer(0)
    shp_spatial_ref = lyr.GetSpatialRef()
    proj_name2 = shp_spatial_ref.GetAttrValue("projcs")

    print(proj_name)
    print(proj_name2)

    if proj_name != proj_name2:
        print("error! ")
        raise Exception("proj error")

    # run main
    status = main(process_dict)


if __name__ == "__main__":
    """
    mainly for landsat 8 seperate band tif imgs
    """
    ori_ras_path_list = [
        "/home/tq/tq-data03/landsat_sr/LC08/01/118/029/LC08_L1TP_118029_20180811_20180815_01_T1/",
        "/home/tq/tq-data05/landsat_sr/LC08/01/114/028/LC08_L1TP_114028_20180831_20180912_01_T1/",
        "/home/tq/tq-data05/landsat_sr/LC08/01/117/027/LC08_L1TP_117027_20180820_20180829_01_T1/",
        "/home/tq/tq-data05/landsat_sr/LC08/01/119/027/LC08_L1TP_119027_20180818_20180829_01_T1/",
        "/home/tq/tq-data05/landsat_sr/LC08/01/115/028/LC08_L1TP_115028_20180806_20180814_01_T1/",
        "/home/tq/tq-data03/landsat_sr/LC08/01/119/028/LC08_L1TP_119028_20180802_20180814_01_T1/",
        "/home/tq/tq-data03/landsat_sr/LC08/01/115/027/LC08_L1TP_115027_20180806_20180814_01_T1/",
        "/home/tq/tq-data03/landsat_sr/LC08/01/119/024/LC08_L1TP_119024_20180802_20180814_01_T1/",
    ]
    for f in ori_ras_path_list:
        go_main(f)

    print("fin")

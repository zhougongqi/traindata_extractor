import os
import glob
import pprint
import settings
from common import logger

from traindata_extractor.ground_truth.TrainDataExtractor import TrainDataExtractorV2
from traindata_extractor.model.RunPredictor import run_predictor

from traindata_extractor.general.common import *
from traindata_extractor.general.Vividict import Vividict
from traindata_extractor.model.SvmModel import SVMClassifier
from traindata_extractor.model.skModel import skModel
from traindata_extractor.model.DecisionTree import DecisionTree

# test model
from sklearn.neural_network import MLPClassifier
# from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn import svm
from sklearn.metrics import confusion_matrix
import json
import numpy as np


def load_json(json_file_path):
    with open(json_file_path, "r") as fp:
        tmp = json.load(fp)
    return tmp


def main(process_dict: dict, model_path):

    process_dict["model_path"] = model_path

    print(process_dict)
    status, result_path = run_predictor(process_dict)

    pprint.pprint(process_dict)

    # run predictor

    print("fin")

    return True


if __name__ == "__main__":
    """
    mainly for landsat 8 seperate band tif imgs
    """
    home = os.path.expanduser('~')
    crop = 'corn'
    ori_ras_path = home + "/tq-data05/landsat_sr/LC08/01/126/031/LC08_L1TP_126031_20180904_20180912_01_T1/"
    # "/home/tq/tq-data04/landsat_sr/LC08/01/120/031/LC08_L1TP_120031_20180809_20180815_01_T1/"
    pro_ras_list = [
        # "/home/tq/tq-data04/landsat_sr/LC08/01/120/030/LC08_L1TP_120030_20180910_20180913_01_T1/"
        # home + "/tq-data05/landsat_sr/LC08/01/119/026/LC08_L1TP_119026_20180818_20180829_01_T1/",
        # home + "/tq-data05/landsat_sr/LC08/01/126/031/LC08_L1TP_126031_20180819_20180829_01_T1/",
        home + "/tq-data05/landsat_sr/LC08/01/126/031/LC08_L1TP_126031_20180904_20180912_01_T1/"
    ]
    process_dict = {
        "img_pro_dict": {},
        # "shp_reproj_dict": {
        #     "samples": "/home/tq/data_pool/china_crop/Liaoning/shape/roi_2c_RiOt_L8_shp_all_utm_n51_3tile_2.shp"
        # },
        "pro_ras_list": pro_ras_list,
        "work_path": home + "/data_pool/U-TMP/NJ/out/",
        # "field_name": "label",
        "traindata_path_npy": home + "/data_pool/U-TMP/NJ/npys/TD_all_corn_other_soybeans_ALL_v1.npy",
        "read_order_list": home + "/data_pool/U-TMP/NJ/npys/TD_LC08_L1TP_119025_corn_v1_ro.json",
    }
    # glob wanted files
    img_pro_dict, band_num = get_bands_into_a_dict(ori_ras_path, "*sr_band*.tif")
    process_dict["img_pro_dict"] = img_pro_dict

    models = {
        'corn': home + '/data_pool/U-TMP/NJ/out/mlp/corn/MLPClassifier_test20181206180254.m',
        'soybeans': home + '/data_pool/U-TMP/NJ/out/mlp/soybeans/MLPClassifier_test20181207174238.m',
        'rice': home + '/data_pool/U-TMP/fujin/out/mlp/rice/MLPClassifier_test20181204180938.m',
    }
    model = models[crop]
    # run main
    status = main(process_dict, model)

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


def main(process_dict: dict, lab: int):
    pprint.pprint(process_dict)
    # img_prepro_dict = process_dict["img_pro_dict"]

    # get a read order list
    read_order_list = load_json(process_dict["read_order_list"])
    pprint.pprint(read_order_list)

    traindata = np.load(process_dict["traindata_path_npy"])


    ## train model
    # svm = SVMClassifier(traindata, process_dict["work_path"])
    # svm.fit()

    # dt_max_depth = 15
    # dt_crossValidation_num = 10
    # model_path = DecisionTree(
    #     traindata,
    #     process_dict["work_path"],
    #     max_depth=dt_max_depth,
    #     crossValidation_num=dt_crossValidation_num,
    # )

    # train model
    # RandomForestClassifier
    # MLPClassifier

    skm = skModel(MLPClassifier(), traindata, process_dict["work_path"], label=lab)
    model_path = skm.fit()

    process_dict["model_path"] = model_path

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
    reg = 'NJ'
    crop = 'corn'
    iid = 1
    pro_ras_list = [
        home + "/tq-data05/landsat_sr/LC08/01/119/025/LC08_L1TP_119025_20180818_20180829_01_T1/",
        # home + "/tq-data05/landsat_sr/LC08/01/119/026/LC08_L1TP_119026_20180818_20180829_01_T1/"
    ]
    npy_path = home + "/data_pool/U-TMP/" + reg + "/npys/"
    all_npy = "TD_all_corn_soybeans_rice_other_ALL_v1.npy"
    order_json = "TD_LC08_L1TP_119025_20180818_corn_v1_ro.json"
    process_dict = {
        "img_pro_dict": {},
        "pro_ras_list": pro_ras_list,
        "work_path": home + "/data_pool/U-TMP/" + reg + "/out/mlp/" + crop + '/',
        "traindata_path_npy": npy_path + all_npy,
        "read_order_list": npy_path + order_json,
    }

    if not os.path.exists(process_dict["work_path"]):
        os.makedirs(process_dict["work_path"])
    # run main
    status = main(process_dict, iid)

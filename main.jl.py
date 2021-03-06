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
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier


def main(process_dict: dict):
    pprint.pprint(process_dict)
    img_prepro_dict = process_dict["img_pro_dict"]

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
    skm = skModel(MLPClassifier(), traindata, process_dict["work_path"])
    model_path = skm.fit()
    process_dict["model_path"] = model_path

    status, result_path = run_predictor(process_dict)

    pprint.pprint(process_dict)

    # run predictor

    print("fin")

    return True


if __name__ == "__main__":
    """
    deal with jilin and heilongjiang
    """
    ori_ras_path = "/home/tq/tq-data04/landsat_sr/LC08/01/120/031/LC08_L1TP_120031_20180809_20180815_01_T1/"
    # "/home/tq/tq-data04/landsat_sr/LC08/01/120/031/LC08_L1TP_120031_20180809_20180815_01_T1/"
    pro_ras_list = load_json(
        # "/home/tq/data_pool/china_crop/Jilin-Heilongjiang/JL_result_2018-test.json"
        "/home/tq/data_pool/china_crop/Jilin-Heilongjiang/JL_result_2018.json"
    )
    # ("/home/tq/data_pool/X-EX/china/JL/JL_result_2018.json")
    process_dict = {
        "img_pro_dict": {},
        "shp_reproj_dict": {
            "samples": "/home/tq/data_pool/china_crop/Liaoning/shape/roi_2c_RiOt_L8_shp_all_utm_n51_3tile_2.shp"
        },
        "pro_ras_list": pro_ras_list,
        "work_path": "/home/tq/data_pool/china_crop/Jilin-Heilongjiang/out/mlp5/",
        "field_name": "label",
        "traindata_path_npy": "/home/tq/data_pool/china_crop/Jilin-Heilongjiang/npys/TD_all_114-028-20180831v5.npy_115-027-20180806v5.npy_115-028-20180806v5.npy_117-027-20180820v5.npy_118-029-20180811v5.npy_119-024-20180802v5.npy_119-027-20180818v5.npy_119-028-20180802v5.npy_v5.npy",
        "read_order_list": "/home/tq/data_pool/china_crop/Liaoning/npys/TD_118-032-20180827_ro.json",
    }
    # glob wanted files
    img_pro_dict, band_num = get_bands_into_a_dict(ori_ras_path, "*sr_band*.tif")
    process_dict["img_pro_dict"] = img_pro_dict

    # run main
    status = main(process_dict)

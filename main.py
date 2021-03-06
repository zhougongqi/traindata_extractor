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
    mainly for landsat 8 seperate band tif imgs
    """
    ori_ras_path = "/home/tq/tq-data04/landsat_sr/LC08/01/120/031/LC08_L1TP_120031_20180809_20180815_01_T1/"
    # "/home/tq/tq-data04/landsat_sr/LC08/01/120/031/LC08_L1TP_120031_20180809_20180815_01_T1/"
    pro_ras_list = [
        # "/home/tq/tq-data04/landsat_sr/LC08/01/120/030/LC08_L1TP_120030_20180910_20180913_01_T1/"
        "/home/tq/tq-data04/landsat_sr/LC08/01/120/031/LC08_L1TP_120031_20180809_20180815_01_T1/",
        "/home/tq/tq-data05/landsat_sr/LC08/01/121/031/LC08_L1TP_121031_20180816_20180829_01_T1/",
        "/home/tq/tq-data04/landsat_sr/LC08/01/120/030/LC08_L1TP_120030_20180910_20180913_01_T1/",
        "/home/tq/tq-data04/landsat_sr/LC08/01/120/031/LC08_L1TP_120031_20180809_20180815_01_T1/",
        "/home/tq/tq-data03/landsat_sr/LC08/01/120/032/LC08_L1TP_120032_20180825_20180829_01_T1/",
        "/home/tq/tq-data04/landsat_sr/LC08/01/120/033/LC08_L1TP_120033_20180825_20180829_01_T1/",
        "/home/tq/tq-data03/landsat_sr/LC08/01/119/030/LC08_L1TP_119030_20180802_20180814_01_T1/",
        "/home/tq/tq-data05/landsat_sr/LC08/01/119/031/LC08_L1TP_119031_20180802_20180814_01_T1/",
        "/home/tq/tq-data03/landsat_sr/LC08/01/119/032/LC08_L1TP_119032_20180802_20180814_01_T1/",
        "/home/tq/tq-data05/landsat_sr/LC08/01/118/031/LC08_L1TP_118031_20180827_20180830_01_T1/",
        "/home/tq/tq-data05/landsat_sr/LC08/01/118/032/LC08_L1TP_118032_20180827_20180830_01_T1/",
        "/home/tq/tq-data05/landsat_sr/LC08/01/118/033/LC08_L1TP_118033_20180827_20180830_01_T1/",
    ]
    process_dict = {
        "img_pro_dict": {},
        "shp_reproj_dict": {
            "samples": "/home/tq/data_pool/china_crop/Liaoning/shape/roi_2c_RiOt_L8_shp_all_utm_n51_3tile_2.shp"
        },
        "pro_ras_list": pro_ras_list,
        "work_path": "/home/tq/data_pool/china_crop/Liaoning/out/mlp2/",
        "field_name": "label",
        "traindata_path_npy": "/home/tq/data_pool/china_crop/Liaoning/npys/TD_all_118-032-20180827v2.npy_119-030-20180802v2.npy_120-032-20180825v2.npy_v2.npy",
        "read_order_list": "/home/tq/data_pool/china_crop/Liaoning/npys/TD_118-032-20180827_ro.json",
    }
    # glob wanted files
    img_pro_dict, band_num = get_bands_into_a_dict(ori_ras_path, "*sr_band*.tif")
    process_dict["img_pro_dict"] = img_pro_dict

    # run main
    status = main(process_dict)

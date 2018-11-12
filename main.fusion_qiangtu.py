import os
import glob
import pprint
import settings
from common import logger

from traindata_extractor.ground_truth.TrainDataExtractor import TrainDataExtractorV2
from traindata_extractor.model.RunPredictor import run_predictor

from traindata_extractor.general.common import *
from traindata_extractor.util.combine_with_qiangtu import *


# test model
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier


def main(combine_dict: dict):
    """
    fusion mlp output with qiangtu result to fill the gap of clouds
    """
    pprint.pprint(combine_dict)

    status = combine_with_qiangtu(combine_dict)

    # pprint.pprint(combine_dict)

    # run predictor

    print("fin")

    return True


if __name__ == "__main__":
    """
    deal with results fusion
    """
    #
    combine_dict = {
        "results_path": "/home/tq/data_pool/china_crop/Jilin-Heilongjiang/out/mlp5/",
        "qiangtu_path": "/home/tq/data_pool/cleanup/waterfall_data/crop_tif/Landsat/20180401-20180930/china-rice-e3-2018/",
        "qa_path": "/home/tq/data_pool/china_crop/Jilin-Heilongjiang/JL_result_2018.json",
        "work_path": "/home/tq/data_pool/china_crop/Jilin-Heilongjiang/out/combined_by_qiangtu3/",
        "tmp_path": "/home/tq/data_pool/china_crop/Jilin-Heilongjiang/out/tmp/",
    }

    # run main
    status = main(combine_dict)

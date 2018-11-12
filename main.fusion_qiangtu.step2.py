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


if __name__ == "__main__":
    """
    deal with results fusion
    """
    #
    z_path = "/home/tq/data_pool/china_crop/Jilin-Heilongjiang/out/fusion/fusion_result_J.tif"
    q_path = "/home/tq/data_pool/china_crop/Jilin-Heilongjiang/out/fusion/qiangtu-jilin"
    # "/home/tq/data_pool/china_crop/Jilin-Heilongjiang/out/fusion/qiangtu-heilongjiang"
    # "/home/tq/data_pool/china_crop/Jilin-Heilongjiang/out/fusion/qiangtu-jilin"
    work_path = "/home/tq/data_pool/china_crop/Jilin-Heilongjiang/out/fusion/1/"

    # run main
    status = combine_with_qiangtu_step2(z_path, q_path, work_path)

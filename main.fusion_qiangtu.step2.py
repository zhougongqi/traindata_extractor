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
    after merging in arcgis and cutted by provinces' shape and reprojection,
    finally, this program will stack result and qiangtu to match them pixel to pixel,
    and replace the cloud pixels with qiangtu's pixels.
    (1,2) will be restored to (0,1)
    and morphological of dilate will be applied on qiangtu because qiangtu's quality is
    poor by now.

    """
    # my result's path
    z_path = "/home/tq/data_pool/china_crop/Jilin-Heilongjiang/out/fusion/fusion_result_J.tif"
    # qiangtu's path
    q_path = "/home/tq/data_pool/china_crop/Jilin-Heilongjiang/out/fusion/qiangtu-jilin"
    # "/home/tq/data_pool/china_crop/Jilin-Heilongjiang/out/fusion/qiangtu-heilongjiang"
    # "/home/tq/data_pool/china_crop/Jilin-Heilongjiang/out/fusion/qiangtu-jilin"

    # output path
    work_path = "/home/tq/data_pool/china_crop/Jilin-Heilongjiang/out/fusion/2/"

    # run main
    status = combine_with_qiangtu_step2v2(z_path, q_path, work_path)

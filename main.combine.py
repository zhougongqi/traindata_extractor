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

import numpy as np

if __name__ == "__main__":
    reg = 'NJ'
    ori_ras_path = (
        "/home/zy/data_pool/U-TMP/" + reg + "/npys/"
    )  # TD_118-032-20180827.npy

    # glob wanted files
    npylist = get_bands_into_a_list(ori_ras_path, "TD_LC08_L1TP_" + "*v1.npy")
    npylist = [x for x in npylist if 'all' not in x]
    pprint.pprint(npylist)

    all_npy = combine_npys(npylist)

    # correct cloud data to label 6
    idx = np.where(all_npy[:, :7] == [-0.9999]*7)[0]
    all_npy[idx, 7] = 6.0
    print(all_npy.shape)
    print(all_npy[:10])

    outpath = ori_ras_path + "TD_all_"
    for n in npylist:
        npylabel = os.path.basename(n).split("_")[-2]
        outpath = outpath + npylabel + "_"
    outpath = ori_ras_path + 'TD_all_corn_soybeans_rice_other_ALL_'
    outpath = outpath + "v1.npy"
    np.save(outpath, all_npy)


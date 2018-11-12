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


if __name__ == "__main__":

    ori_ras_path = (
        "/home/tq/data_pool/china_crop/Jilin-Heilongjiang/npys/"
    )  # TD_118-032-20180827.npy

    # glob wanted files
    npylist = get_bands_into_a_list(ori_ras_path, "*v5.npy")

    all_npy = combine_npys(npylist)
    print(all_npy.shape)

    outpath = ori_ras_path + "TD_all_"
    for n in npylist:
        npylabel = os.path.basename(n).split("_")[1]
        outpath = outpath + npylabel + "_"
    outpath = outpath + "v5.npy"

    all_npy = all_npy.astype(np.float)
    all_npy[:, 0:7] = all_npy[:, 0:7] / 10000.0

    # binarize labels
    indx = np.where(all_npy == 2.)
    all_npy[indx] = 0

    indx = np.where(all_npy == 3.)
    all_npy[indx] = 0

    indx = np.where(all_npy == 4.)
    all_npy[indx] = 0

    print(all_npy[100:105, :])
    np.save(outpath, all_npy)


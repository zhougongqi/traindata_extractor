import os
import glob
import pprint
import settings
from common import logger

from traindata_extractor.ground_truth.TrainDataExtractor import TrainDataExtractorV2

from traindata_extractor.general.common import *
from traindata_extractor.general.Vividict import Vividict


def main(process_dict: dict):
    pprint.pprint(process_dict)
    img_prepro_dict = process_dict["img_pro_dict"]

    # get a read order list
    read_order_list, feat_list_1d = set_vvdic_key_order(img_prepro_dict)
    pprint.pprint(read_order_list)

    #
    tde = TrainDataExtractorV2(
        process_dict, read_order_list, sample_label="l8_heishan_test"
    )
    status = tde.set_join_char("_")
    traindata, feat_name_list, npypath = tde.go_get_mask_2npy()
    print("fin")

    return True


if __name__ == "__main__":
    ori_ras_path = "/home/tq/tq-data04/landsat_sr/LC08/01/120/031/LC08_L1TP_120031_20180809_20180815_01_T1/"
    process_dict = {
        "img_pro_dict": {},
        "shp_reproj_dict": {
            "samples": "/home/tq/data_pool/zgq/crop_class/rois/roi_2c_RiOt_L8_shp_all_utm_n51.shp"
        },
        "work_path": "/home/tq/data_pool/zgq/crop_class/test/",
        "field_name": "label",
    }
    # glob wanted files
    filelist = glob.glob(ori_ras_path + "*sr_band*.tif")
    img_name = ori_ras_path.split("/")[-2]
    img_pro_dict = Vividict()
    bn = 0
    for f in filelist:
        bn += 1
        band_str = "band_" + str(bn)
        img_pro_dict[img_name][band_str] = filelist[bn - 1]
    process_dict["img_pro_dict"] = img_pro_dict

    # run main
    status = main(process_dict)

import os
import glob
import pprint
import settings
from common import logger

from traindata_extractor.general.common import *
from traindata_extractor.ground_truth.TrainDataExtractor import TrainDataExtractorV2


def main(process_dict: dict, iid):
    """
    this is a program process only traindata extraction
    """
    pprint.pprint(process_dict)
    img_prepro_dict = process_dict["img_pro_dict"]

    # get a read order list
    read_order_list, feat_list_1d = set_vvdic_key_order(img_prepro_dict)
    process_dict["read_order_list"] = read_order_list
    pprint.pprint(read_order_list)

    #
    tde = TrainDataExtractorV2(
        process_dict, read_order_list, sample_label="l8_fujin_test", label=iid,
    )

    traindata, feat_name_list, npypath = tde.go_get_mask_2npy()
    process_dict["traindata_path_npy"] = npypath
    print("fin")

    return True


if __name__ == "__main__":
    """
    mainly for landsat 8 seperate band tif imgs
    """
    home = os.path.expanduser('~')
    reg = 'NJ'
    utm = '51N'
    crops = ['corn', 'soybeans', 'rice', 'other']
    iids = [1, 2, 4, 6]
    ori_ras_path = home + "/tq-data05/landsat_sr/LC08/01/119/026/LC08_L1TP_119026_20180818_20180829_01_T1/"
    tile = ori_ras_path.split('_')[3:5]
    work_path = home + "/data_pool/U-TMP/" + reg + "/npys/"
    assert len(crops) == len(iids)
    for i in range(len(crops)):
        crop = crops[i]
        iid = iids[i]
        outname_label = 'LC08_L1TP_{}_{}_{}_'.format(tile[0], tile[1], crop)

        process_dict = {
            "outname_label": outname_label + "v1",
            "img_pro_dict": {},
            "shp_reproj_dict": {
                'samples': home + "/data_pool/U-TMP/" + reg + "/shp/" + crop + "_" + utm + ".shp",
            },
            "work_path": work_path,
            "field_name": "id",
        }

        # glob wanted files
        img_pro_dict, band_num = get_bands_into_a_dict(ori_ras_path, "*sr_band*.tif")
        img_pro_list = get_bands_into_a_list(ori_ras_path, "*sr_band*.tif")
        process_dict["img_pro_dict"] = img_pro_dict
        # print(process_dict)

        # run main
        status = main(process_dict, iid)

    npy_file = os.listdir(work_path)
    npy_file = [x for x in npy_file if '.npy' in x]
    print('Extracted sample files: ')
    pprint.pprint(npy_file)



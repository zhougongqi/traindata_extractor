import numpy as np

import logging
from traindata_extractor.general.common import get_reverse_dict
from traindata_extractor.general.common import data_shuffle_col
from traindata_extractor.general.Vividict import Vividict
from traindata_extractor.general.feat_calc import feat_calc
from traindata_extractor.general.common import (
    load_json,
    add_root_path,
    delete_999_row,
    delete_nan_row,
)

"""
temporary useless in this project
"""


class TrainDataProcessor:
    my_logger = logging.getLogger(__qualname__)

    def __init__(
        self,
        process_dict: dict,
        aux_dic: dict,
        npy_path_list: list,
        read_order_list: list,
    ):
        """
        Function:
            initialize TrainDataProcessor class
        Input:
            npy_path_list: a list contains the paths of npy files.
            raster_paths:a 2-layered dict contains the raster files
                        as the model input.
                        like this:
                        {"sensor-1":{"band-1":"band path",
                                     "band-2":"band path"],
                                    ...}
                         "sensor-2":{"band-1":"band path",
                                     "band-2":"band path",
                                    ...}
                        }
            label_dict:  a dictionary contains label of each ground-truth
                        polygon, a field named "label" is necessary.
            bandmath_list: a 2-layered list contains the band math info,
                        like this:
                        [["addition",0,1],
                         ["division",2,3],...]
            read_order_list: 2d list of read order
        """
        # set class members
        self.npy_path_list = npy_path_list

        img_prepro_dict = load_json(process_dict["img_pro_dict"])
        shp_prepro_dict = load_json(process_dict["shp_reproj_dict"])
        img_prepro_dict, shp_prepro_dict = add_root_path(
            img_prepro_dict, shp_prepro_dict
        )
        self.work_path = process_dict["work_path"]
        self.label_dict = aux_dic["label_dict"]
        self.raster_path_dict = img_prepro_dict
        self.command = process_dict["command"]
        self.bandmath_list = self.command["band_math"]
        self.feat_list = [f[1] for f in read_order_list]
        self.read_order_list = read_order_list
        self.n_files = len(self.npy_path_list)
        self.calc_symbols = {
            "addition": "+",
            "subtraction": "-",
            "multiplication": "*",
            "division": "/",
        }  # zzz todo: add set_calc_symbols func();

        # get the reversed dict
        self.label_dict_R = get_reverse_dict(self.label_dict)

        # load first npy
        self.data = np.load(self.npy_path_list[0]).item()

        # statistic the dict in npy
        self.statistic_data()

        # set a default relabel dict for classification
        self.set_relabel_dict()

        # set a default proportion dict
        self.set_proportion_dict()

        # set feature dictionary using feat_list
        self.feat_dict = {}
        n = 0
        for l in self.feat_list:
            self.feat_dict[l] = n
            n += 1

    def set_label_dicts(self, dic: dict):
        """
        Function:
            set self.__label_dict and label_dict_R
            if you have a new label-dictionary.
        """
        self.label_dict = dic
        self.label_dict_R = get_reverse_dict(dic)
        self.my_logger.info("label dictionary updated!~")

    def statistic_data(self):
        """
        Function:
            statistic self.data, get level 1,2,3 keys in kxl list.
            and set them as class members.
            self.feat_dict is dict of feature name and feature colomn index
            in future ndarrays.
        """
        k1l = []
        k2l = []
        k3l = []
        feat_dic = {}
        k3n = 0
        for k1 in self.data.keys():
            k1l.append(k1)
        for k2 in self.raster_path_dict.keys():
            k2l.append(k2)
            for k3 in self.raster_path_dict[k2].keys():
                k3l.append(k3)
                feat_dic[k3] = k3n
                k3n += 1
        self.k1_list = k1l
        self.k2_list = k2l
        self.k3_list = k3l
        # self.feat_dict = feat_dic
        n_pix = 0
        for k1 in self.data.keys():
            n_pix += len(self.data[k1][k2l[0]][k3l[0]])
        self.n_pixel = n_pix

    def set_relabel_dict_bylist(self, list1: list):
        """
        Function:
            set new_label_dict by a given list $list1
            $list1 contains the labels to be set to 1
        """
        for key in self.new_label_dict.keys():
            self.new_label_dict[key] = 0
        for items in list1:
            self.new_label_dict[items] = 1
        self.my_logger.info("new label dictionary is set!")

    def set_relabel_dict(self, new_dic: dict = None):
        """
        Function:
            set new_label_dict by a given dict $new_dic
            re-label means combine original class into several new classes,
                and set some new label to the combined classes.
        """
        if new_dic is None:  # default new dict, only palm is 1
            new_dic = self.label_dict_R
            for lab in new_dic.keys():
                new_dic[lab] = 0
                if lab == "palm":
                    new_dic[lab] = 1
            self.new_label_dict = new_dic
            self.my_logger.info("default re-label-dict is set")
        else:
            self.new_label_dict = new_dic
            self.my_logger.info("new re-label-dict is set")

    def set_proportion_dict(self, new_dic: dict = None) -> (np.array, dict):
        """
        Function:
            set proportion_dict by a given dict new_dic
                proportion_dict has the same keys as self.label_dict,
                it stores the proportion of each class when applying them
                to classifier. this will help control some class has
                too large samples that may cause unexpected training results.
        """
        if new_dic is None:  # default new dict, all proportion is 1 (no sampling)
            new_dic = self.label_dict_R
            for lab in new_dic.keys():
                new_dic[lab] = 1
            self.proportion_dict = new_dic
            self.my_logger.info("default proportion-dict is set")
        else:
            self.proportion_dict = new_dic
            self.my_logger.info("new proportion-dict is set")

    def get_feat_name(self, bm_list: list) -> str:
        """
        get a feature name string from bandmath command list
        """
        b1 = bm_list[1]  # band 1 name
        b2 = bm_list[2]  # band 2 name
        if b1.find("+") or b1.find("-"):
            b1 = "(" + b1 + ")"
        if b2.find("+") or b2.find("-"):
            b2 = "(" + b2 + ")"

        feat_str = b1 + self.calc_symbols[bm_list[0]] + b2
        return feat_str

    def dict_to_nparray_new(self) -> (np.ndarray, dict):
        """
        Function:
            get nparray-like traindata from dict
            first seperate each category into a old-version-like dict,
            then adjust the category amount by a certain dict.
        Input:
            deal with only class members.
        Output:
            array: a ndarray contains all the pixels and all the features and
            all the new label numbers, like this:
            feat1   feat2   feat3 ...   label
            ---------------array contains below----------------------
            xxx.xx  xxx.xx  xxx.xx ...  1
            xxx.xx  xxx.xx  xxx.xx ...  0
            xxx.xx  xxx.xx  xxx.xx ...  0
            xxx.xx  xxx.xx  xxx.xx ...  1
            xxx.xx  xxx.xx  xxx.xx ...  0

            dict: a dictionary contains feature keys and corresponding colomn
            indices.

        ###### NOTICE!!! ######
            before the band math, array is transposed and all processes
            is based on the transposed array!
        """

        fn = len(self.feat_list)  # feature number
        pn = self.n_pixel  # sample points(pixels) number
        array = np.zeros((fn + 1, pn))  # feature number + 1 label row
        array[:, :] = -999  # np.nan
        cate_dic = Vividict()
        for c in self.label_dict_R:  # category
            for f in self.feat_list:  # feature list
                cate_dic[c][f] = []
            cate_dic[c]["label"] = []
        # loop npy dic
        for k1 in self.data.keys():
            cate = self.data[k1]["category"]
            for ro in self.read_order_list:
                k2 = ro[0]
                k3 = ro[1]
                sam = self.data[k1][k2][k3]
                cate_dic[cate][k3] = np.hstack((cate_dic[cate][k3], sam))
                lab = np.zeros_like(sam)
                lab[:] = self.label_dict_R[cate]
                # change labels to newlabel according to newlabel_dict
                lab[:] = self.new_label_dict[cate]
            cate_dic[cate]["label"] = np.hstack((cate_dic[cate]["label"], lab))

        # print length of each item
        all_sam_num = 0
        for c in self.label_dict_R:  # category
            for f in self.feat_list:  # feature
                print(c, ":", f, ":", len(cate_dic[c][f]))
                all_sam_num += len(cate_dic[c][f])

        if all_sam_num == 0:
            self.my_logger.info("dict_to_nparray_new(): no traindata found!")
            return None

        print("---resampling---")
        array = np.array([])
        for cate1 in cate_dic.keys():
            cate_feat = np.array([])
            pp = self.proportion_dict[cate1]  # proportion
            assert 0 < pp <= 1, "proportion must be in (0,1]"
            for feat1 in cate_dic[cate1].keys():
                if len(cate_dic[cate1][feat1]) == 0:
                    continue
                if cate_feat.size > 0:
                    cate_feat = np.vstack([cate_feat, cate_dic[cate1][feat1]])
                else:
                    cate_feat = cate_dic[cate1][feat1]
                n_sam = len(cate_dic[cate1][feat1])
            # data shuffle and sampling
            if len(cate_dic[cate1][feat1]) == 0:
                continue
            cate_feat = data_shuffle_col(cate_feat)
            s_sam = int(n_sam * pp)
            cate_feat = cate_feat[:, 0:s_sam]
            print("%s : %d", cate1, len(cate_feat[0, :]))
            if array.size > 0:
                array = np.hstack([array, cate_feat])
            else:
                array = cate_feat

        self.my_logger.info("array shuffling...")
        array = data_shuffle_col(array)
        array = array.swapaxes(1, 0)

        # band math
        n_feat = len(self.feat_dict)
        if self.bandmath_list is not None:  # run band math
            self.my_logger.info("applying band math ......")
            for bm in self.bandmath_list:  # for each band math command
                if type(bm) is list:
                    self.my_logger.info(
                        "calculating "
                        + bm[0]
                        + " of bands: {} , {}".format(bm[1], bm[2])
                    )
                    array = feat_calc(
                        array, self.feat_dict[bm[1]], self.feat_dict[bm[2]], bm[0]
                    )
                    n_feat += 1
                    feat_str = self.get_feat_name(bm)
                    self.feat_dict[feat_str] = n_feat - 1
                else:
                    self.my_logger.error("wrong bandmath_list format!")
                    return None, None
        else:
            self.my_logger.error("bandmath_list is empty!")
            return None, None

        array = delete_999_row(array)
        array = delete_nan_row(array)
        self.data = array

        td_name = self.work_path + "td_all_label.npy"
        np.save(td_name, self.data)
        self.my_logger.info("{}".format(self.feat_dict))
        return array, self.feat_dict

    def multi_dicts_to_nparray(self) -> np.ndarray:
        """
        Function:
            get nparray-like traindata from self.npy_path_list
        Input:
            deal with only class members.
        Output:
            array: a ndarray contains all the pixels and all the features and
            all the new label numbers
        """
        if self.n_files < 2:
            self.my_logger.info(
                "there is only one file in list, execute single version instead"
            )
            td_all, feat_dic0 = self.dict_to_nparray_new()
            return True

        td_all = np.array([])
        # loop each file in list
        for nf in self.npy_path_list:
            self.data = np.load(nf)
            self.statistic_data()
            td, feat_dic0 = self.dict_to_nparray_new()
            td_all = np.vstack([td_all, td])

        return td_all

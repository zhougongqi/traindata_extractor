import logging
import math
import os
import numpy as np

from osgeo import gdal
from osgeo import ogr
from osgeo import osr

from traindata_extractor.general.Vividict import Vividict
from traindata_extractor.general.common import *
from traindata_extractor.general.calc_mask_by_shape import (
    calc_mask_by_shape,
    calc_mask_by_shape_block,
)


class TrainDataExtractorV2:
    my_logger = logging.getLogger(__qualname__)

    def __init__(
        self,
        process_dic: dict,
        read_order_list: list,
        *,
        label_field_name: str = "label",
        label_keep_list: list = [1],
        is_binarize: bool = True,
        poly_id_field: str = "type",
        bound_field: str = "ID_1",
        mask_value: int = 1,
        place_label: str = "OV",
        time_label: str = "201709",
        sample_label: str = "test",
    ):
        """
        Function:
            initialize TrainDataExtractor class
        Input:
            raster_dic:a 2-layered dict contains the raster files
                       as the model input.
                       like this:
                        {"sensor-1":{"band-1":"band path",
                                     "band-2":"band path"],
                                    ...}
                         "sensor-2":{"band-1":"band path",
                                     "band-2":"band path",
                                    ...}
                        }
            sampe_shp_path: training shape file path, a file that contains
                        all the ground-truth polygons drawn by human.
            bound_path:  boundary shape file of entire research area, eg:
                        a country shp-file in state/province level.
            work_path:   a path that stores temporary files or results.
            label_dict:  a dictionary contains label of each ground-truth
                        polygon, a field named "label" is necessary.
            read_order_list: 2d list of read order
            aux_data:   a auxilary data dictionay, including band_list and its
                        invalid value info.

            *: optional parameters:
            field_name:  field name specified for mask making, "label" for
                        default.
            mask_value:  default mask value for mask-making, 1 for default;
            place_label: a string for final .npy file naming. means place
                        name.
            time_label:  a string for final .npy file naming. means img time.
            sample_label: a string for final .npy file naming. means the
                        ground-truth file label.
        """
        # set dictionary variables
        img_dict = process_dic["img_pro_dict"]
        shp_dict = process_dic["shp_reproj_dict"]
        img_dict, shp_dict = add_root_path(img_dict, shp_dict)
        img_dict = transform_inner_to_vvdic(img_dict)

        self.img_dict = img_dict
        self.shp_dict = shp_dict
        self.shp_label_path = shp_dict["samples"]
        self.work_path = process_dic["work_path"]
        self.outname_label = process_dic["outname_label"]

        # key para
        self.field_name = process_dic["field_name"]
        self.mask_value = mask_value
        self.label_keep_list = label_keep_list
        self.ref_coef = 10000
        self.isbinarize = is_binarize

        self.read_order_list = read_order_list
        self.__n_block = 1

        # set joint char that connects each part in output filename
        self.__join_char = "_"

        # star symbols rotating while extracting traindata, pretending the
        # program is still alive...
        self.str_arrs = ["~", "/", "|", "\\"]

        self.sample_label = sample_label

    def set_join_char(self, char: str) -> bool:
        """
        set the join char if you don't like "_"
        """
        if len(char) != 1:
            self.my_logger.error("set_join_char(): char length not 1!")
            return False
        if char in ["\\", "|", "/", ":", "*", "?", '"', "<", ">", " "]:
            self.my_logger.error("set_join_char(): illegal char!")
            return False
        self.__joinchar = char
        return True

    def set_keep_label(self, klist: list) -> bool:
        self.label_keep_list = klist
        return True

    def set_block_num(self, num: int):
        """
        set blocking number
        """
        self.__n_block = num
        self.my_logger.info("block number has set to {}".format(num))

    def set_label_dicts(self, dic: dict):
        """
        Function:
            set self.__label_dict and __label_dict_R if you have a
            new label-dictionary.
        """
        self.__label_dict = dic
        self.__label_dict_R = get_reverse_dict(dic)
        self.my_logger.info("label dictionary updated!~")

    def check_proj(self, shape_path: str, ras_path: str) -> bool:
        """
        Function:
            check the projection of shapefile and raster,
            if inconsist, raise an exception.
        Input:
            shape_path: path of shapefile
            ras_path:   path of raster image
        Output:
            No output. just continue if no exception is thrown.
        """
        # open shapefile
        try:
            shapef = ogr.Open(shape_path)
            lyr = shapef.GetLayer(0)
            shp_spatial_ref = lyr.GetSpatialRef()
        except Exception as e:
            self.my_logger.error("open shape file error: %s", shape_path)
            return False

        # open rasterfile
        try:
            ds = gdal.Open(ras_path)
            ras_proj = ds.GetProjection()
            ras_spatial_ref = osr.SpatialReference(wkt=ras_proj)
        except Exception as e:
            self.my_logger.error("open raster file error: %s", ras_path)
            return False

        if shp_spatial_ref.GetAttrValue("projcs") != ras_spatial_ref.GetAttrValue(
            "projcs"
        ):
            self.my_logger.error("check_proj(): incorrect projection, check again!~")
            return False
        else:
            self.my_logger.info("projection check passed!")
        return True

    def get_ori_data(self, data_path: str, band_id: int) -> np.ndarray:
        """
        Function:
            get raster data in ndarray from file
        Input:
            data_path: path of a single band raster file.
        Output:
            oriData: ndarray contains the required data.
        """
        try:
            dataset = gdal.Open(data_path)
            oriData = dataset.GetRasterBand(band_id).ReadAsArray()
        except Exception as e:
            self.my_logger.error("open raster file error: %s", data_path)
            return None
        return oriData

    def get_valid_data(
        self,
        data: np.ndarray,
        mask: np.ndarray,
        *,
        nodata_value: list = [0],
        validRange: list = [0, 10000],
    ) -> (np.ndarray, np.ndarray):
        """
        Function:
            old version - may have memory crushes
            get valid ground-truth data by applying mask
        Input:
            data:   ndarray of satellite image.
            mask:   ndarray of mask.
            *: optional parameters:
            nodata_value: values to be ignored in $ data.
            iss1:   boolean, if satellite is Sentinel-1, True; else, False.
            validRange:
                    valid range of values in $data.
        Output:
            train_data: flattened array contains the required train data.
            mask_t:     flattened array contains the mask(label value).
        """
        mask_idx = np.where(mask > 0)
        train_data = data[mask_idx]
        mask_t = mask[mask_idx]
        if train_data.shape == mask_t.shape:
            pass  # print(train_data.shape)
        else:
            raise Exception("get_valid_data_new(): shape not match! skip")

        # replace nodata value to nan
        for n in range(len(nodata_value)):
            train_data[train_data == nodata_value[n]] = -999

        return train_data.flatten(), mask_t.flatten()  # train_data[:, np.newaxis]

    def get_valid_data_new(
        self,
        data: np.ndarray,
        mask: np.ndarray,
        *,
        nodata_value: list = [0],
        validRange: list = [0, 10000],
        n_block: int = 10,
    ) -> (np.ndarray, np.ndarray):
        """
        Function:
            new version, a little bit slow
            get valid ground-truth data by applying mask
        Input:
            data:   ndarray of satellite image.
            mask:   ndarray of mask.
            *: optional parameters:
            nodata_value:
                    values to be ignored in $data.
            iss1:   boolean, if satellite is Sentinel-1, True; else, False.
            validRange:
                    valid range of values in $data.
            n_block: num of block
        Output:
            train_data: flattened array contains the required train data.
            mask_t:     flattened array contains the mask(label value).
        """
        x, y = data.shape
        # get quantile break points on y
        delta_Y = int(math.floor(y / n_block))
        perc = list(range(0, y + delta_Y, delta_Y))
        perc.pop()
        perc[-1] = y

        train_data_all = np.array([])
        mask_t_all = np.array([])

        # read valid data by block
        for i in range(n_block):
            tmp_start = perc[i]
            tmp_end = perc[i + 1]
            mask0 = mask[:, tmp_start:tmp_end]
            mask_id = mask[:, tmp_start:tmp_end]
            data0 = data[:, tmp_start:tmp_end]
            mask0 = mask0 * 1.0
            mask0[mask0 > 0] = 1.0
            mask0[mask0 <= 0] = np.nan
            valid_data = np.multiply(data0, mask0)
            train_data = valid_data[~np.isnan(valid_data)]
            mask_t = mask_id[~np.isnan(valid_data)]
            train_data = train_data / 1.0

            if train_data.shape == mask_t.shape:
                # print(train_data.shape)
                pass
            else:
                self.my_logger.error("get_valid_data_new(): shape not match! skip")
                return None, None

            # replace nodata value to nan
            for n in range(len(nodata_value)):
                train_data[train_data == nodata_value[n]] = np.nan

            train_data_all = np.hstack([train_data_all, train_data.flatten()])
            mask_t_all = np.hstack([mask_t_all, mask_t.flatten()])

        return train_data_all, mask_t_all  # train_data[:, np.newaxis]

    def binarize_label(self):
        # label keep list is given
        labels = self.label_keep_list
        new_label = self.train_label.copy()
        new_label[:] = 0
        for lb in labels:
            idx = np.where(self.train_label == lb)
            new_label[idx] = 1

        return new_label

    def stat_labels(self):
        labels_list = []
        for lb in self.train_label:
            if lb not in labels_list:
                labels_list.append(lb)
        self.unique_label_list = labels_list
        print("num of labels: {}, \nlabels: {}".format(len(labels_list), labels_list))

    def feature_norm_ref(self):
        """
        normalize features to reflectance (devide by 10000)
        """
        self.train_feature = self.train_feature * 1.0 / self.ref_coef

    def go_get_mask_2npy(self) -> (np.ndarray, list, str):
        """
        Function:
            directly put raster data into array
        Input:
        Output:
            ndarray contains all features and label, like this:

                feat1   feat2   ... featn   label
                --------------------------------data is below
                xxx.xx  xxx.xx  ... xxx.xx  0
                xxx.xx  xxx.xx  ... xxx.xx  1
                ...
            npypath:      saved npy file path.

        NOTICE: train feature and train label are transposed before they are combined
                at #zzz_transpose!
        """

        # get geo information from the first item in dict
        try:
            first_raster_name = next(self.img_dict.walk())[-1]
            ds = gdal.Open(first_raster_name)
            geo_trans = ds.GetGeoTransform()
            x_size = ds.RasterXSize
            y_size = ds.RasterYSize
            img_shape = [x_size, y_size]
        except Exception as e:
            self.my_logger.error("open raster file error: {}".format(first_raster_name))
            return None, None, None

        # making label ma
        self.my_logger.info("Generating label mask ...")
        mask, num_label, list_label = calc_mask_by_shape(
            self.shp_label_path,
            geo_trans,
            img_shape,
            specified_field=self.field_name,
            condition=None,
            mask_value=-1,
            flag_dlist=True,
            field_strict=True,
        )
        if mask is None:
            self.my_logger.error("calculating label mask error")
            return None, None, None
        print("There are {} polygons".format(num_label))

        # get raster data
        self.my_logger.info("Getting raster data...")
        # traindata_dict = Vividict()
        td_arr = None
        feat_name_list = []
        n_file = len(self.read_order_list)
        fn = 0

        # loop read order list, read all raster datas
        for ro in self.read_order_list:
            fn += 1
            print("{}/{} raster files:".format(fn, n_file))
            k1 = ro[0]
            k2 = ro[1]
            ds = gdal.Open(self.img_dict[k1][k2])
            n_bands = ds.RasterCount
            feat_str = os.path.basename(self.img_dict[k1][k2])

            for b in range(n_bands):
                bn = b + 1  # band_id, starts from 1
                print("-- {}/{} band:".format(bn, n_bands))
                if n_bands >= 2:  # add a band string to feat_str
                    band_str = "b" + str(bn)
                    feat_str1 = feat_str + band_str
                else:
                    band_str = ""
                    feat_str1 = feat_str

                # read raster data band bn
                ras_data = self.get_ori_data(self.img_dict[k1][k2], bn)
                if ras_data is None:
                    self.my_logger.error("get ori data error")
                    return None, None, None

                # get valid data with mask
                # traindata is a flattened 1d array
                traindata, pids = self.get_valid_data(ras_data, mask)
                if traindata is None:
                    self.my_logger.error("get valid data error")
                    return None, None

                self.my_logger.info(
                    "getting data from ["
                    + k1
                    + "] ["
                    + k2
                    + "] {}".format(traindata.shape)
                )

                # append data to td_arr
                if td_arr is None:
                    td_arr = traindata
                    print("    ->", td_arr.shape)
                else:
                    td_arr = np.vstack((td_arr, traindata))
                    print("    ->", td_arr.shape)
                # write featname into a list
                feat_name_list.append(feat_str1)

        self.train_label = pids
        self.train_feature = td_arr

        # process data
        # self.feature_norm_ref() #zzz
        self.stat_labels()
        if self.isbinarize:
            pids = self.binarize_label()
        else:
            pids = pids

        # add labels at the last colomn
        td_arr = np.vstack((self.train_feature, pids))
        print("add label:")
        print("    ->", td_arr.shape)

        # zzz_transpose!
        td_arr = td_arr.T
        td_arr = data_shuffle(td_arr)
        print("transposed    ->", td_arr.shape)
        print(feat_name_list)

        # zzz todo: add invalid value removal
        # delete all zero cols
        td_arr = delete_0s_row(td_arr)
        td_arr = delete_999_row(td_arr)

        npy_path = self.work_path + "TD_" + self.outname_label + ".npy"
        np.save(npy_path, td_arr)

        rolist_path = npy_path.replace(".npy", "_ro.json")
        save_json(rolist_path, self.read_order_list)
        self.my_logger.info("save traindata success!")
        self.my_logger.info(npy_path)

        return td_arr, feat_name_list, npy_path

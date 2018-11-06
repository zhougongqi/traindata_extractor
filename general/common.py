import json
import os
import sys
import glob
import math
from osgeo import gdalnumeric
from PIL import Image
import numpy as np
from traindata_extractor.general.Vividict import Vividict


def image_to_array(i: Image) -> np.ndarray:
    """
    Converts a Python Imaging Library array to a
    gdalnumeric image.
    """
    a = gdalnumeric.fromstring(i.tobytes(), dtype=np.int32)
    a.shape = i.im.size[1], i.im.size[0]
    return a


def image_to_array_byte(i: Image) -> np.ndarray:
    """
    Converts a Python Imaging Library array to a
    gdalnumeric image.
    """
    a = gdalnumeric.fromstring(i.tobytes(), dtype=np.int8)
    a.shape = i.im.size[1], i.im.size[0]
    return a


def array_to_image(a: np.ndarray) -> Image:
    """
    Converts a gdalnumeric array to a
    Python Imaging Library Image.
    """
    i = Image.frombytes("L", (a.shape[1], a.shape[0]), (a.astype(np.int32)).tobytes())
    return i


def array_to_image_byte(a: np.ndarray) -> Image:
    """
    Converts a gdalnumeric array to a
    Python Imaging Library Image.
    """
    i = Image.frombytes("I", (a.shape[1], a.shape[0]), (a.astype(np.int32)).tobytes())
    return i


def world_to_pixel(geo_matrix: tuple, x: int, y: int) -> tuple:
    """
    Uses a gdal geomatrix (gdal.GetGeoTransform()) to calculate
    the pixel location of a geospatial coordinate
    """
    up_left_x = geo_matrix[0]
    up_left_y = geo_matrix[3]
    x_dist = geo_matrix[1]
    pixel = int((x - up_left_x) / x_dist)
    line = int((up_left_y - y) / x_dist)
    return pixel, line


def pixels_to_world(geo_matrix: tuple, pixel: int, line: int) -> tuple:
    """
    Uses a gdal geomatrix (gdal.GetGeoTransform()) to calculate
    the geo-location of a img pixel coordinate
    """
    up_left_x = geo_matrix[0]
    up_left_y = geo_matrix[3]
    x_dist = geo_matrix[1]
    x = pixel * x_dist + up_left_x
    y = up_left_y - line * x_dist
    return x, y


def placelist_to_numeric(plist: list, state_id: dict, state_id_group: dict) -> list:
    """
    Function:
        turn input string list to numeric list
    Input:
        plist: string list
        state_id: dictionary from string to numeric
                like this:
                {"pulau":1
                 "kedah":2,...}
        state_id_group: dictionary from minus digit(group) to positive
                like this:
                {-1:[2,3,4]
                 -2:[4,5,6],...}
    Output:
        an int list.
    """
    nlist = []
    for p in plist:
        if p in state_id.keys():
            if state_id[p] > 0:  # if it is a state
                nlist.append(state_id[p])
            else:  # if it is a group
                nlist.extend(state_id_group[state_id[p]])
        else:
            raise Exception(
                "placelist_to_numeric(): key not found in state dictionary!"
            )
    # remove repeated elements
    olist = []
    for i in nlist:
        if i not in olist:
            olist.append(i)
    return olist


def get_reverse_dict(dic: dict) -> dict:
    """
    Function:
        reverse the dictionary
    Input:
        dic: dictionary to be reversed
    Output:
        rdic: reversed dict, keys are values in $dic,
                and values are keys in $dic.
    """
    rdic = {}
    for k, v in dic.items():
        rdic.setdefault(v, k)
    return rdic


def data_shuffle(data: np.ndarray) -> np.ndarray:
    """
    shuffle 2-D data along rows
    """
    idx = np.arange(len(data))
    np.random.shuffle(idx)
    data = data[idx, :]
    return data


def data_shuffle_col(data: np.ndarray) -> np.ndarray:
    """
    shuffle 2-D data along colomns
    """
    row, col = data.shape
    idx = np.arange(col)
    np.random.shuffle(idx)
    data = data[:, idx]
    return data


def delete_nan_col(array: np.ndarray) -> np.ndarray:
    """
    Function:
        delete the colomns contains NaN values in given array $array
    return:
        a new array that deleted some colomns where contains NaN.
    """
    # row is feature, so process by each row
    row, col = array.shape
    arr = array
    for r in range(row):
        idx_nan = np.argwhere(np.isnan(arr[r, :]))
        idx_nan = idx_nan.flatten()
        if len(idx_nan):
            arr = np.delete(arr, idx_nan, axis=1)
    print(array.shape, "NaN deleted!, new shape is :", arr.shape)
    return arr


def delete_nan_row(array: np.ndarray) -> np.ndarray:
    """
    Function:
        delete the rows contains NaN and Inf values in given array $array
    return:
        a new array that deleted some rows where contains NaN.
    """
    # row is feature, so process by each row
    row, col = array.shape
    arr = array
    for r in range(col):
        idx_nan = np.argwhere(np.isnan(arr[:, r]))
        idx_nan = idx_nan.flatten()
        if len(idx_nan):
            arr = np.delete(arr, idx_nan, axis=0)

    row, col = arr.shape
    for r in range(col):
        idx_nan = np.argwhere(np.isinf(arr[:, r]))
        idx_nan = idx_nan.flatten()
        if len(idx_nan):
            arr = np.delete(arr, idx_nan, axis=0)

    row, col = arr.shape
    for r in range(col):
        idx_nan = np.argwhere(np.isneginf(arr[:, r]))
        idx_nan = idx_nan.flatten()
        if len(idx_nan):
            arr = np.delete(arr, idx_nan, axis=0)
    print(array.shape, "NaN, Inf, -Inf deleted!, new shape is :", arr.shape)
    return arr


def delete_999_row(array: np.ndarray) -> np.ndarray:
    """
    Function:
        delete the rows contains NaN values in given array $array
    return:
        a new array that deleted some rows where contains NaN.
    """
    # row is feature, so process by each row
    row, col = array.shape
    arr = array
    for r in range(col):
        idx_nan = np.where(arr == -999)
        # idx_nan = idx_nan.flatten()
        if len(idx_nan):
            arr = np.delete(arr, idx_nan, axis=0)
    print(array.shape, "-999 deleted!, new shape is :", arr.shape)
    return arr


def replace_invalid_value(array: np.ndarray, new_value: int) -> np.ndarray:
    """
    Function:
        replace the  NaN, Inf, -Inf values in given array $array
    return:
        a new array without NaN.
    """
    where_are_nan = np.isnan(array)
    array[where_are_nan] = new_value

    where_are_inf = np.isinf(array)
    array[where_are_inf] = new_value

    where_are_isneginf = np.isneginf(array)
    array[where_are_isneginf] = new_value
    return array


def set_vvdic_key_order(vv: Vividict) -> list:
    """
    set feature read order in 2-D list for a 2-D dict
    """
    ro_list = []
    feat_list_1d = []
    n = 0
    for k1 in vv.keys():
        for k2 in vv[k1].keys():
            ro_list.append([k1, k2, n])
            feat_list_1d.append(k2)  # just the 2nd layer keys: features
            n += 1
    return ro_list, feat_list_1d


def get_traindata_key_dict(ro_list: list) -> dict:
    """
    get a dict which feature name is key, colomn number is value
    """
    traindata_key_dic = {}
    for ro in ro_list:
        # ro is like this ["sentinel", "VV", 0]
        #                   ^ this is sensor name
        #                               ^ this is feature name
        #                                    ^ this is read/colomn order
        traindata_key_dic[ro[1]] = int(ro[2])
    return traindata_key_dic


def transform_inner_to_vvdic(dic: dict) -> Vividict:  # , dic:dict
    """
    transform a dict to vividict all layers
    """
    dicv = dic
    for key, value in dicv.items():  # dic.items():
        if type(value) is dict:
            dicv[key] = transform_inner_to_vvdic(value)
    return Vividict(dicv)


def save_json(json_file_path, file_dict):
    with open(json_file_path, "w") as fp:
        json.dump(file_dict, fp, ensure_ascii=True, indent=2)


def load_json(json_file_path):
    with open(json_file_path, "r") as fp:
        tmp = json.load(fp)
    return tmp


def add_root_path(img_pro_dict: dict, shp_reproj_dict: dict) -> (dict, dict):
    """
    Function:
        path add root for path, such as "data_pool/test" to "/home/tq/data_pool/test"
    """
    home_dir = os.path.expanduser("~")
    img_pro_dict = {
        key: {k: os.path.join(home_dir, v) for k, v in img_pro_dict[key].items()}
        for key in img_pro_dict.keys()
    }
    shp_reproj_dict = {k: os.path.join(home_dir, v) for k, v in shp_reproj_dict.items()}
    # add home dir to each file path
    return img_pro_dict, shp_reproj_dict


def get_bands_into_a_dict(img_path: str):
    filelist = glob.glob(img_path + "*sr_band*.tif")
    filelist.sort()
    img_name = img_path.split("/")[-2]
    img_pro_dict = Vividict()
    bn = 0
    for f in filelist:
        bn += 1
        band_str = "band_" + str(bn)
        img_pro_dict[img_name][band_str] = filelist[bn - 1]
    return img_pro_dict, bn


def print_progress_bar(now_pos: int, total_pos: int):
    n_sharp = math.floor(50 * now_pos / total_pos)
    n_space = 50 - n_sharp
    sys.stdout.write(
        "  ["
        + "#" * n_sharp
        + " " * n_space
        + "]"
        + "{:.2%}\r".format(now_pos / total_pos)
    )


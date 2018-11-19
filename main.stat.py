#!/home/tq/anaconda3/bin/python
# -*- coding:utf-8 -*-
import os
import glob
import pprint

# import settings
from common import logger

from traindata_extractor.general.common import *
from traindata_extractor.general.shp_pro_conv import *
from traindata_extractor.general.validation import *


if __name__ == "__main__":
    """
    stat each city/county's crop area.
    """
    # my result's path
    cls_path = "/home/tq/data_pool/china_crop/Jilin-Heilongjiang/out/fusion/2/fusion_result_J_fusion_final2.tif"
    # "/home/tq/data_pool/china_crop/Jilin-Heilongjiang/out/fusion/2/fusion_result_L_rep.tif"

    # shape path
    shp_path = (
        "/home/tq/data_pool/china_crop/vector/province-with-county/jilin-county-15.shp"
        # "/home/tq/data_pool/china_crop/vector/province-with-city/liaoning-city.shp"
    )

    work_path = os.path.dirname(shp_path)
    home_dir = os.path.expanduser("~")

    prj_path = "/home/tq/data_pool/china_crop/Jilin-Heilongjiang/out/china_wgs84.prj"

    # shape reproj
    shp_reproj_path = shp_path.replace(".shp", ".reproj.shp")
    new_shp_path = shp_pro_conv(shp_path, work_path, prj_path, home_dir)

    # run main
    status = stat_area_zonal_county15(cls_path, new_shp_path)
    # stat_area_zonal_county(cls_path, new_shp_path)
    # stat_area_zonal(cls_path, new_shp_path)

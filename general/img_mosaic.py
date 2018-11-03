import subprocess
import os
import gdal
import logging

my_logger = logging.getLogger(__name__)


def img_mosaic(
    input_img_list: str,
    dst_img_folder: str,
    res_name: str,
    prj_path: str,
    home_dir: str,
    *,
    pixel_size: float = None,
    src_nodata: list = None,
    dst_nodata: list = None,
    res_label: str = "",
) -> str:
    """
    Function:
        mosaic multi-images to one big tif-image, all images will be re-projected
    Input:
        input_img_list: string, pathes of the input image files
        dst_img_folder: string, folder path where the results are saved,
                        including intermediate files
        res_name: string, result file names, including vrt file and result tif image
        prj_path: string, path of the projection file used to reproject all images
        home_dir: expanduser path
        pixel_size: float, optional (default = None)pixel size of final tif image,
                    if not given, consistent with the input images
        src_nodata: (float) list, optional (default = None)
                    pixel value of nodata in source image,
                    if not given, consistent with the input images
        dst_nodata: (float) list, optional (default = None)
                    pixel value of nodata in vrt,
                    if not given, consistent with the input images
        res_label: string, label in result file name for marking
    Output:
        img_res_path: string, path of result tif image,
                      if is None, save result image failed
    """
    # add user path to input_img_list and dst_img_folder
    input_img_list = [os.path.join(home_dir, img_path) for img_path in input_img_list]

    # check if result folder is existed
    assert os.path.exists(dst_img_folder), "dst image folder must be existed"

    # remove duplicate element
    input_img_list = list(set(input_img_list))

    # Parameter settings in the command line (pixel size / src nodata / dst nodata)
    pixel_size_str = (
        "" if pixel_size is None else "-tr " + str(pixel_size) + " " + str(pixel_size)
    )

    src_nodata_str = (
        ""
        if src_nodata is None
        else "-srcnodata '{0}'".format(
            " ".join(map(str, src_nodata))
        )  # "-srcnodata nodata1,nodata2,nodata3..."
    )
    dst_nodata_str = (
        ""
        if dst_nodata is None
        else "-dstnodata '{0}'".format(
            " ".join(map(str, dst_nodata))
        )  # "-dstnodata nodata1,nodata2,nodata3..."
    )

    # projection conversion
    my_logger.info("projection conversion...")
    prj_img_list = []
    index = 0
    for ori_img_path in input_img_list:
        index = index + 1
        ori_img_name = os.path.splitext(os.path.basename(ori_img_path))[0]
        prj_img_path = os.path.join(
            dst_img_folder,
            ori_img_name + "_prj_{0}_{1}.tif".format(res_label, str(index)),
        )
        prj_cmd_str = (
            "gdalwarp -t_srs "
            + prj_path
            + " -r cubic "
            + src_nodata_str
            + " "
            + dst_nodata_str
            + " -of GTiff -overwrite "
            + ori_img_path
            + " "
            + prj_img_path
        )
        my_logger.info("projection conversion command %s", prj_cmd_str)
        process_status = subprocess.run(prj_cmd_str, shell=True)

        # check process status
        if process_status.returncode != 0:
            my_logger.error("projection conversion failed!")
            return None
        prj_img_list.append(prj_img_path)

    my_logger.success("projection conversion success!")

    if len(input_img_list) == 1:
        return prj_img_list[0].replace(home_dir + os.sep, "")

    # build vrt
    my_logger.info("build vrt...")
    vrt_res_path = os.path.join(dst_img_folder, res_name) + "_" + res_label + ".vrt"
    vrt_cmd_str = (
        "gdalbuildvrt "
        + src_nodata_str
        + " "
        + dst_nodata_str.replace("dstnodata", "vrtnodata")
        + " "
        + vrt_res_path
        + " "
        + " ".join(prj_img_list)
        + " -overwrite"
    )
    my_logger.info("build vrt command %s", vrt_cmd_str)
    # "nodata1,nodata2,nodata3" -> "nodata1 nodata2 nodata3" gdalbuildvrt command
    # does not support ','
    process_status = subprocess.run(vrt_cmd_str, shell=True)

    # check process status
    if process_status.returncode != 0:
        my_logger.error("build vrt failed!")
        return None

    my_logger.success("build vrt success!")

    # get mosaic results
    my_logger.info("image mosaic...")
    img_res_path = os.path.join(dst_img_folder, res_name) + "_" + res_label + ".tif"
    img_cmd_str = (
        "gdal_translate "
        + pixel_size_str
        + " -r cubic -of GTiff "
        + vrt_res_path
        + " "
        + img_res_path
    )
    my_logger.info("image mosaic command %s", img_cmd_str)
    process_status = subprocess.run(img_cmd_str, shell=True)

    # check process status
    if process_status.returncode != 0:
        my_logger.error("image mosaic failed!")
        return None

    # delete reproj image files
    for prj_img_file in prj_img_list:
        try:
            os.remove(prj_img_file)
        except Exception as e:
            my_logger.error(
                "delete reproject image file failed: {0}".format(prj_img_file)
            )
            return None

    my_logger.info("reproject image file deleted")

    my_logger.success("image mosaic success!")

    ds = gdal.Open(img_res_path)
    return img_res_path.replace(home_dir + os.sep, "") if ds else None

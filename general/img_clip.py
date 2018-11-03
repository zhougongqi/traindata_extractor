import subprocess
import os
import ogr
import gdal
import logging

my_logger = logging.getLogger(__name__)


def img_clip(
    shp_file_path: str,
    src_img_path: str,
    dst_img_folder: str,
    home_dir: str,
    *,
    src_nodata: list = None,
    dst_nodata: list = None,
    pixel_size: float = None,
    field_name: str = None,
    field_condition: list = None,
) -> list:
    """
    Function:
        clip image by shape file, both polygon and multi-polygon can be used
    Input:
        shp_file_path: string, path of the shape file
        src_img_path: string, path of the image file
        dst_img_folder: string, folder path of the result image(s)
        home_dir: expanduser path
        src_nodata: (float) list, optional (default = None),
                    pixel value of nodata in source image,
                    if not given, consistent with the input image
        dst_nodata: (float) list, optional (default = None),
                    pixel value of nodata in destination image,
                    if not given, consistent with the input image
        pixel_size: float, optional (default = None)
                    pixel size of result tif image,
                    if not given, consistent with the input image
        field_name: string, optional (default = None)
                    A field name of shape to determine which polygons were used to clip,
                    if not given, clip by the ROI
        field_condition: (string) list, optional (default = None)
                    if the value of field_name is in field_condition,
                    the polygon was chosed to clip
    Output:
        img_res_path: (string) list, path list of result tif image(s),
                      if is None, save result image failed
    """

    # Get shape parameters
    shp_ds = ogr.Open(shp_file_path)
    shp_lyr = shp_ds.GetLayer(0)
    shp_lyr.ResetReading()
    ori_shp_name = os.path.splitext(os.path.basename(shp_file_path))[0]
    ori_img_name = os.path.splitext(os.path.basename(src_img_path))[0]
    img_res_path = []

    # Parameter settings in the command line (pixel size / src nodata / dst nodata)
    pixel_size_str = (
        "" if pixel_size is None else "-tr " + str(pixel_size) + " " + str(pixel_size)
    )
    src_nodata_str = (
        ""
        if src_nodata is None
        else '-srcnodata "{0}"'.format(
            ",".join(map(str, src_nodata))
        )  # "-srcnodata nodata1,nodata2,nodata3..."
    )
    dst_nodata_str = (
        ""
        if dst_nodata is None
        else '-dstnodata "{0}"'.format(
            ",".join(map(str, dst_nodata))
        )  # "-dstnodata nodata1,nodata2,nodata3..."
    )

    # clip image by the ROI
    if field_name is None:
        # rename dst tif file
        dst_tif_name = ori_img_name + "_clipBy_" + ori_shp_name + ".tif"
        dst_tif_path = os.path.join(dst_img_folder, dst_tif_name)

        # command line for image clip
        clip_cmd_str = (
            "gdalwarp "
            + pixel_size_str
            + " -r cubic"  # cubic resampling method
            + " --config GDALWARP_IGNORE_BAD_CUTLINE YES "
            + src_nodata_str
            + " "
            + dst_nodata_str
            + " -of GTiff"
            + " -cutline {0}".format(shp_file_path)
            + " -crop_to_cutline -overwrite "
            + src_img_path
            + " "
            + dst_tif_path
        )
        my_logger.info("clip command: %s", clip_cmd_str)

        # "nodata1,nodata2,nodata3" -> "nodata1 nodata2 nodata3"
        # gdalbuildvrt command does not support ','
        process_status = subprocess.run(
            [sub_str.replace(",", " ") for sub_str in clip_cmd_str.split()]
        )

        # check process status
        if process_status.returncode != 0:
            my_logger.error("Clip image failed!")
            return None

        ds = gdal.Open(dst_tif_path)
        if ds is None:
            my_logger.error("Save file error:", dst_tif_path)
            return None
        else:
            img_res_path.append(dst_tif_path.replace(home_dir + os.path.sep, ""))

    # clip image by each inner polygon, if field_condition is given, clip the
    # specific polygon only
    else:
        shp_ft = shp_lyr.GetNextFeature()
        while shp_ft:
            field_value = shp_ft.GetField(field_name)
            my_logger.info("%s: %s", field_name, field_value)
            shp_ft = shp_lyr.GetNextFeature()
            # choose polygen based on field condition
            if (field_condition is not None) and (field_value not in field_condition):
                continue

            # rename dst tif file
            dst_tif_name = (
                ori_img_name
                + "_clipBy_"
                + ori_shp_name
                + "_"
                + str(field_value).replace(" ", "_")
                + ".tif"
            )
            dst_tif_path = os.path.join(dst_img_folder, dst_tif_name)

            # command line for image clip
            clip_cmd_str = (
                "gdalwarp "
                + pixel_size_str
                + " -r cubic"  # cubic resampling method
                + " --config GDALWARP_IGNORE_BAD_CUTLINE YES "
                + src_nodata_str
                + " "
                + dst_nodata_str
                + " -of GTiff"
                + " -cutline {0}".format(shp_file_path)
                + " -cwhere {0}='{1}'".format(field_name, field_value)
                + " -crop_to_cutline -overwrite "
                + src_img_path
                + " "
                + dst_tif_path
            )

            my_logger.info("clip command: %s", clip_cmd_str)

            # "nodata1,nodata2,nodata3" -> "nodata1 nodata2 nodata3" gdalbuildvrt
            # command does not support ','
            process_status = subprocess.run(
                [sub_str.replace(",", " ") for sub_str in clip_cmd_str.split()]
            )

            # check process status
            if process_status.returncode != 0:
                my_logger.error("Clip image failed!")
                return None
            ds = gdal.Open(dst_tif_path)
            if ds is None:
                my_logger.error("Save file error:", dst_tif_path)
                return None
            else:
                img_res_path.append(dst_tif_path.replace(home_dir + os.path.sep, ""))

    my_logger.success("Clip image success!")

    return img_res_path


if __name__ == "__main__":
    shp_file_path = "/home/xyz/data_pool/zgq/vector/malay.shp"
    src_img_path = "/home/xyz/data_pool/Palm/palm_NDVI/201712/EVI_malay_201712.tif"
    dst_img_folder = "/home/xyz/data_pool/Eric/"
    dst_nodata = [0.0, 1.0]
    src_nodata = [0.0, 1.0]
    pixel_size = 500
    field_name = "NAME_1"
    field_condition = ["Trengganu"]
    img_clip(
        shp_file_path,
        src_img_path,
        dst_img_folder,
        src_nodata=src_nodata,
        dst_nodata=dst_nodata,
        pixel_size=pixel_size,
        field_name=field_name,
        field_condition=field_condition,
    )

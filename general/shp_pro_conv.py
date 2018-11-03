import os
import ogr
import osr
import logging

my_logger = logging.getLogger(__name__)


def shp_pro_conv(src_shp_file: str, work_path: str, prj_path: str, home_dir: str):
    """
    Function:
        reproject a single shapefile based on a projection file
    Input:
        src_shp_file: string, path of the original shape file
        dst_shp_file: string, path of the reproject shape file
        prj_path: string, path of the projection file
        home_dir: expanduser path
    Output:
        reprj_shp_file: string, path of the original shape file
    """
    ori_shp_name = os.path.splitext(os.path.basename(src_shp_file))[0]
    file_extent_name = os.path.splitext(os.path.basename(src_shp_file))[1]

    prj_file_name = os.path.splitext(os.path.basename(prj_path))[0]

    reprj_shp_file = os.path.join(
        work_path,
        ori_shp_name + "_reproj_{0}{1}".format(prj_file_name, file_extent_name),
    )
    driver = ogr.GetDriverByName("ESRI Shapefile")

    # get the input spatial info
    try:
        input_data_set = driver.Open(src_shp_file)
    except Exception as e:
        my_logger.error("open source shape error!")
        return None

    input_layer = input_data_set.GetLayer()
    input_spatial_ref = input_layer.GetSpatialRef()

    # get the output spatial info
    output_spatial_ref = osr.SpatialReference()
    try:
        prj_file = open(prj_path, "r")
    except Exception as e:
        my_logger.error("open projection file error!")
        return None

    prj_txt = prj_file.read()
    output_spatial_ref.ImportFromWkt(prj_txt)

    # create the CoordinateTransformation
    sr_trans = osr.CoordinateTransformation(input_spatial_ref, output_spatial_ref)

    # create the output layer
    if os.path.exists(reprj_shp_file):
        driver.DeleteDataSource(reprj_shp_file)
    try:
        output_data_set = driver.CreateDataSource(reprj_shp_file)
    except Exception as e:
        my_logger.error("create output shape error!")

    output_layer = output_data_set.CreateLayer(
        "reproject", output_spatial_ref, geom_type=ogr.wkbMultiPolygon
    )

    # add fields
    input_layer_defn = input_layer.GetLayerDefn()
    for i in range(0, input_layer_defn.GetFieldCount()):
        field_defn = input_layer_defn.GetFieldDefn(i)
        output_layer.CreateField(field_defn)

    # get the output layer's feature definition
    output_layer_defn = output_layer.GetLayerDefn()

    # loop through the input features
    input_feature = input_layer.GetNextFeature()
    while input_feature:
        # get the input geometry
        geom = input_feature.GetGeometryRef()
        # reproject the geometry
        geom.Transform(sr_trans)

        # create a new feature
        output_feature = ogr.Feature(output_layer_defn)
        # set the geometry and attribute
        output_feature.SetGeometry(geom)
        for i in range(0, output_layer_defn.GetFieldCount()):
            output_feature.SetField(
                output_layer_defn.GetFieldDefn(i).GetNameRef(),
                input_feature.GetField(i),
            )
        # add the feature to the shapefile
        output_layer.CreateFeature(output_feature)
        # dereference the features and get the next input feature
        output_feature = None
        input_feature = input_layer.GetNextFeature()

    # Save and close the shapefiles
    input_data_set = None
    output_data_set = None

    my_logger.success("shape reproject success!")
    return reprj_shp_file


if __name__ == "__main__":
    home_dir = os.path.expanduser("~")
    input_shp_dict = {
        "label": home_dir + "/Desktop/IND_Kal.shp",
        "boudary": home_dir + "/data_pool/snap_aux/palm_shape/IND_KL.shp",
    }
    work_path = home_dir + "/data_pool/zgq/test_vortex/"
    prj_path = home_dir + "/data_pool/Ray_EX/PRJ_FILE/palm_wgs84.prj"
    shp_pro_conv(input_shp_dict, work_path, prj_path)

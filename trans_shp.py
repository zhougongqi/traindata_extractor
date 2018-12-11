import shapefile
import gdal
import osr
import ogr
import os
import fiona
from fiona.crs import from_string
from shapely.geometry import mapping, Polygon
from collections import OrderedDict


def getSRSPair(src):
    # 获取投影参考系与地理参考系信息
    pros = osr.SpatialReference()
    pros.ImportFromWkt(src.GetProjection())
    geos = pros.CloneGeogCS()
    return pros, geos


def latlon2geo(dataset, lon, lat):
    pros, geos = getSRSPair(dataset)
    trans = osr.CoordinateTransformation(geos, pros)
    coords = trans.TransformPoint(lon, lat)
    return coords[:2]


def reproj_shp(shp_file, tif_file, id):
    # tif with projections I want
    tif = gdal.Open(tif_file)

    # shapefile with the from projection
    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSource = driver.Open(shp_file, 1)
    layer = dataSource.GetLayer()

    # set spatial reference and transformation
    sourceprj = layer.GetSpatialRef()
    print(sourceprj)
    targetprj = osr.SpatialReference(wkt=tif.GetProjection())

    transform = osr.CoordinateTransformation(sourceprj, targetprj)

    to_fill = ogr.GetDriverByName("Esri Shapefile")
    ds = to_fill.CreateDataSource("/home/zeito/pyqgis_data/projected.shp")
    outlayer = ds.CreateLayer('', targetprj, ogr.wkbPolygon)
    outlayer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))

    # apply transformation
    i = 0

    for feature in layer:
        transformed = feature.GetGeometryRef()
        transformed.Transform(transform)

        geom = ogr.CreateGeometryFromWkb(transformed.ExportToWkb())
        defn = outlayer.GetLayerDefn()
        feat = ogr.Feature(defn)
        feat.SetField('id', i)
        feat.SetGeometry(geom)
        outlayer.CreateFeature(feat)
        i += 1
        feat = None

    ds = None


def transer_shp(shp_file, tif_file, crop, id, res_file):
    srs = gdal.Open(tif_file)
    ras_proj = srs.GetProjection()
    ras_spatial_ref = osr.SpatialReference(wkt=ras_proj)
    proj_name = ras_spatial_ref.GetAttrValue("projcs")
    proj4 = str(ras_spatial_ref.ExportToProj4())
    print(proj4)

    sf = shapefile.Reader(shp_file)
    shapes = sf.shapes()
    new_polys = []
    k = 0

    # coordinate transform from lat,lon to geo-coord
    for i in range(len(shapes)):
        tp = shapes[i].shapeType
        if tp == 5:
            poly = shapes[i].points
            new_poly = []
            for point in poly:
                geox, geoy = latlon2geo(srs, point[0], point[1])
                new_poly.append((geox, geoy))
            new_polys.append(new_poly)
        else:
            k += 1
            continue
    print('Invalid type of Polygons: ', k)
    print('Num of Polygons: ', len(new_polys))
    # print(new_polys[:5])

    schema = {
        'geometry': 'Polygon',
        'properties': OrderedDict([
            ('id', 'int')
        ]),
    }

    # Write a new ShapeFile and project
    new_shp_file = res_file + crop + '_' + proj_name[-3:] + '.shp'
    # ~/data_pool/U-TMP/NJ/shp/corn_52N.shp
    crs = from_string(proj4)
    print(crs)

    with fiona.open(new_shp_file, 'w', driver='ESRI Shapefile', crs=crs, schema=schema) as c:
        for poly in new_polys:
            poly = Polygon(poly)
            c.write({
                'geometry': mapping(poly),
                'properties': OrderedDict([
                    ('id',  id)
                ]),
            })


if __name__ == "__main__":
    home = os.path.expanduser('~')
    reg = 'NJ'
    crop = 'rice'
    id = 4
    tif_file = home + '/tq-data05/landsat_sr/LC08/01/119/025/LC08_L1TP_119025_20180818_20180829_01_T1/'
    shp_file = home + '/data_pool/china_crop/Label_20181110/ALL_label/all_label_' + crop + '.shp'
    # shp_file = '/home/zy/Desktop/all_label_other.shp'
    tif_list = os.listdir(tif_file)
    tif_list = [x for x in tif_list if 'sr_band' in x and 'tfw' not in x]
    ref_tif = tif_file + tif_list[0]
    result_file = home + "/data_pool/U-TMP/" + reg + "/shp/"
    if not os.path.exists(result_file):
        os.makedirs(result_file)
    transer_shp(shp_file, ref_tif, crop, id, result_file)
    print('{}, {}, {}, Done!'.format(reg, crop, id))

import numpy as np

from rasterio.transform import Affine
from rasterio.windows import Window
import rasterio
from rasterio.warp import reproject, calculate_default_transform
from rasterio.enums import Resampling
from rasterstats import zonal_stats
from shapely.geometry import box


def zonal_stats_for_value(
        raster,
        vectors,
        value=None,
        data_value=None,
        stats="count",
        resolution=1
):
    raster_data = raster.read(1)

    if value is not None and data_value is not None:
        raster_data[raster_data == value] = data_value
        raster_data[raster_data != data_value] = raster.nodata

    z_stats = [
        s[stats] * resolution ** 2 for s in
        zonal_stats(
            vectors=vectors,
            raster=raster_data,
            affine=raster.transform,
            stats=stats,
            nodata=raster.nodata
        )
    ]

    return z_stats


def read_raster(filename, crs):
    reprojected_filename = filename.parent / f"{filename.stem}_reprojected{filename.suffix}"
    reproject_raster(filename, reprojected_filename, new_crs=crs)

    raster = rasterio.open(reprojected_filename)

    return raster


def reproject_raster(in_path, out_path, new_crs):
    with rasterio.open(in_path) as src:
        crs = rasterio.crs.CRS.from_string(new_crs)
        transform, width, height = calculate_default_transform(
            src.crs,
            crs,
            src.width,
            src.height,
            *src.bounds
        )

        kwargs = src.meta.copy()
        kwargs.update({
            'crs': crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(out_path, 'w', **kwargs) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=rasterio.transform,
                src_crs=rasterio.crs,
                dst_transform=transform,
                dst_crs=crs,
                resampling=Resampling.nearest
            )


def compute_area(df, cartesian_system_crs='ESRI:102001'):
    # Reproject in a cartesian system to compute area (squared km)
    area = df.geometry.to_crs(cartesian_system_crs).geometry.area / 10 ** 6

    return area


def register_rasters(raster1, raster2, raster3):
    resolution = raster1.res[0]
    assert raster1.res[0] == raster1.res[1]
    assert raster1.res == raster2.res == raster3.res

    box1 = box(*raster1.bounds)
    box2 = box(*raster2.bounds)
    box3 = box(*raster3.bounds)

    intersection = box1.intersection(box2).intersection(box3)

    transform = Affine(
        resolution,
        0.0,
        intersection.bounds[0],
        0.0,
        -resolution,
        intersection.bounds[3]
    )

    raster1 = _register_raster(raster1, intersection, resolution)
    raster2 = _register_raster(raster2, intersection, resolution)
    raster3 = _register_raster(raster3, intersection, resolution)

    return (raster1, raster2, raster3), transform


def _register_raster(raster, intersection, resolution):
    xmin, ymin, xmax, ymax = raster.bounds

    x0 = intersection.bounds[0] + resolution / 2
    y0 = intersection.bounds[3] - resolution / 2
    y1 = intersection.bounds[1] + resolution / 2
    x1 = intersection.bounds[2] - resolution / 2

    row0 = int((ymax - y0) / resolution)
    row1 = int((ymax - y1) / resolution)

    col0 = int((x0 - xmin) / resolution)
    col1 = int((x1 - xmin) / resolution)

    window = Window(col0, row0, col1 - col0 + 1, row1 - row0 + 1)

    registered = raster.read(1, window=window)

    return registered


def zonal_stats_intersection(
        vectors,
        raster1,
        raster2,
        affine,
        data_values,
        stats="count",
        nodata_value=-999,
        resolution=1
):
    new_raster = np.ones(raster1.shape) * nodata_value
    for data_value in data_values:
        new_raster[np.where((raster1 == data_value) & (raster2 == data_value))] = 1

    z_stats = [
        s[stats] * resolution ** 2 for s in zonal_stats(
        vectors=vectors,
        raster=new_raster,
        affine=affine,
        nodata=nodata_value,
        stats=stats
    )]

    return z_stats, new_raster


def zonal_stats_intersection_gain(
        vectors,
        raster1,
        raster2,
        affine,
        data_values,
        stats="count",
        nodata_value=-999,
        resolution=1
):
    new_raster = np.ones(raster1.shape) * nodata_value
    for data_value in data_values:
        new_raster[np.where((raster1 <= data_value) & (raster2 == data_value))] = 1

    z_stats = [
        s[stats] * resolution ** 2 for s in zonal_stats(
        vectors=vectors,
        raster=new_raster,
        affine=affine,
        nodata=nodata_value,
        stats=stats
    )]

    return z_stats, new_raster

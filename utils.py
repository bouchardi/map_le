import numpy as np

import geopandas
from rasterio.windows import Window
from rasterio.features import shapes
from rasterstats import zonal_stats


def zonal_stats_for_value(
        raster,
        vectors,
        value=None,
        data_value=None,
        stats="count",
):
    raster_data = raster.read(1)

    if value is not None and data_value is not None:
        raster_data[raster_data == value] = data_value
        raster_data[raster_data != data_value] = raster.nodata

    z_stats = [
        s[stats] * (raster.res[0] / 1000) ** 2 for s in
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
    reprojected_filename = filename.parent / f"{filename.stem}_reprojected_{crs}{filename.suffix}"
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


def register_raster(raster, intersection):
    resolution = raster.res[0]

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
        data_values,
        affine,
        resolution,
        stats="count",
        nodata_value=-999,
):
    new_raster = np.ones(raster1.shape) * nodata_value
    new_raster[np.where(
        (np.isin(raster1,  data_values)) &
        (np.isin(raster2, data_values)))
    ] = 1

    z_stats = [
        s[stats] * resolution ** 2 for s in zonal_stats(
        vectors=vectors,
        raster=new_raster,
        affine=affine,
        nodata=nodata_value,
        stats=stats
    )]

    return z_stats, new_raster


def raster_to_vector(filename, crs):
    reprojected_filename = filename.parent / f"{filename.stem}_reprojected_{crs}{filename.suffix}"
    reproject_raster(filename, reprojected_filename, new_crs=crs)

    mask = None
    with rasterio.open(reprojected_filename) as src:
        raster = src.read(1)
        results = [
            {'properties': {'raster_val': v}, 'geometry': s}
            for i, (s, v) in enumerate(
                shapes(raster.astype(np.float32), mask=mask, transform=src.transform)
            )
        ]

    gdf = geopandas.GeoDataFrame.from_features(results)

    return gdf


from rasterio.warp import reproject, Resampling, calculate_default_transform
import rasterio


def coregister(infile, match, outfile):
    """Reproject a file to match the shape and projection of existing raster.

    Parameters
    ----------
    infile : (string) path to input file to reproject
    match : (string) path to raster with desired shape and projection
    outfile : (string) path to output file tif
    """
    # open input
    with rasterio.open(infile) as src:
        # open input to match
        with rasterio.open(match) as match:
            dst_crs = match.crs

            # calculate the output transform matrix
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src.crs,  # input CRS
                dst_crs,  # output CRS
                match.width,  # input width
                match.height,  # input height
                *match.bounds,  # unpacks input outer boundaries (left, bottom, right, top)
            )

        # set properties for output
        dst_kwargs = src.meta.copy()
        dst_kwargs.update({"crs": dst_crs,
                           "transform": dst_transform,
                           "width": dst_width,
                           "height": dst_height,
                           "nodata": src.nodata})
        # open output
        with rasterio.open(outfile, "w", **dst_kwargs) as dst:
            # iterate through bands and write using reproject function
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)
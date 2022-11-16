import rasterio
from rasterio.warp import reproject, calculate_default_transform
from rasterio.enums import Resampling
from rasterstats import zonal_stats


def zonal_stats_for_value(raster, vectors, value, stats, data_value, no_data_value, affine):
    new_raster = raster.copy()
    new_raster[new_raster == value] = data_value
    new_raster[new_raster != data_value] = no_data_value

    z_stats = [
        s[stats] for s in
        zonal_stats(
            vectors=vectors,
            raster=new_raster,
            affine=affine,
            stats=stats,
            nodata=no_data_value
        )
    ]

    return z_stats


def read_raster(filename, crs):
    reprojected_filename = filename.parent / f"{filename.stem}_reprojected.{filename.suffix}"
    reproject_raster(filename, reprojected_filename, new_crs=crs)

    with rasterio.open(reprojected_filename) as src:
        no_data_value = src.nodata
        affine = src.transform

        data = src.read(1)

    return data, affine, no_data_value


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

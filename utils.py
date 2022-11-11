import geopandas
import rasterio
from pathlib import Path

from rasterio.warp import reproject, calculate_default_transform
from rasterio.enums import Resampling


def read_raster(filename):
    with rasterio.open(filename) as src:
        data = src.read(1)

    return data


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

def main():
    data_path = Path("data/")

    data_2011_2040 = read_raster(filename=data_path / "rasters" / "arcp8510000532011-2040.asc")
    data_2041_2070 = read_raster(filename=data_path / "rasters" / "arcp8510000532041-2070.asc")
    data_2071_2100 = read_raster(filename=data_path / "rasters" / "arcp8510000532071-2100.asc")

    boundaries = geopandas.read_file(data_path / "lcar000b21a_e.shp")
    boundaries.plot()


if __name__ == '__main__':
    main()

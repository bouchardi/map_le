import geopandas
import rasterio
from pathlib import Path


def read_raster(filename):
    with rasterio.open(filename) as src:
        data = src.read(1)

    return data


def main():
    data_path = Path("data/")

    data_2011_2040 = read_raster(filename=data_path / "rasters" / "arcp8510000532011-2040.asc")
    data_2041_2070 = read_raster(filename=data_path / "rasters" / "arcp8510000532041-2070.asc")
    data_2071_2100 = read_raster(filename=data_path / "rasters" / "arcp8510000532071-2100.asc")

    boundaries = geopandas.read_file(data_path / "lcar000b21a_e.shp")
    boundaries.plot()


if __name__ == '__main__':
    main()

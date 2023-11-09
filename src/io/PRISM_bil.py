# import gdal
import rioxarray
import rasterio
f = '/Users/felipe/tmp/prism/PRISM_ppt_stable_4kmM3_2022_all_bil'
src = rasterio.open(f)

# def ReadBilFile(bil):
#     gdal.GetDriverByName('EHdr').Register()
#     img = gdal.Open(bil)
#     band = img.GetRasterBand(1)
#     data = band.ReadAsArray()
#     return data

def get_values(unixtime:int,options=None):
    pass
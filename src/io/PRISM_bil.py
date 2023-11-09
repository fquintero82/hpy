# import gdal
import rioxarray
import rasterio
f = '/Users/felipe/tmp/prism/prism/PRISM_ppt_stable_4kmM3_2022_bil.bil'
src = rasterio.open(f)
data = src.read()
# def ReadBilFile(bil):
#     gdal.GetDriverByName('EHdr').Register()
#     img = gdal.Open(bil)
#     band = img.GetRasterBand(1)
#     data = band.ReadAsArray()
#     return data

def get_values(unixtime:int,options=None):
    pass
#import Image as Image
import rasterio
from rasterio.plot import show
from PIL import Image
import rasterio
import geotiff
import tifffile as tiff
import matplotlib.pyplot as plt
import numpy as np
import cv2
'RGB'
img = cv2.imread("C:\\Users\\ma\\project\\Scripts\\data\\lab1\\ShahzadpurHH.tif")
img2 = cv2.imread("C:\\Users\\ma\\project\\Scripts\\data\\lab1\\ShahzadpurHH.tif")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
plt.imshow(img2_rgb)
plt.axis('off')
plt.show()


"canny edge of image"
"""try:
    img = cv2.imread("C:\\Users\\ma\\project\\Scripts\\data\\lab1\\ShahzadpurHH.tif")
    img2 = cv2.imread("C:\\Users\\ma\\project\\Scripts\\data\\lab1\\ShahzadpurHH.tif")
    edges = cv2.Canny(img, 15, 10)
    edges2 = cv2.Canny(img2, 15, 10)
    cv2.imwrite('result.tiff', edges)
    plt.imshow(edges)
    cv2.imwrite('result.tiff', edges2)
    plt.imshow(edges2)
    plt.show()
except IOError:
    print('Error while reading files !!!')"""

"translating the image"
"""M = np.float32([[1, 0, 100], [0, 1, 50]])
try:
    # Read image from disk.
    img = cv2.imread("C:\\Users\\ma\\project\\Scripts\\data\\lab1\\ShahzadpurHH.tif")
    img2 = cv2.imread("C:\\Users\\ma\\project\\Scripts\\data\\lab1\\ShahzadpurHH.tif")
    res = cv2.warpAffine(img, M, (100,50 ))
    res2 = cv2.warpAffine(img2, M, (100, 50))
    cv2.imwrite('result.tiff', res)
    plt.imshow(res)
    cv2.imwrite('result.tiff', res2)
    plt.imshow(res2)
    plt.show()
except IOError:
    print('Error while reading files !!!')"""

'rotating the image'
"""try:
    # Read image from disk.
    img = cv2.imread("C:\\Users\\ma\\project\\Scripts\\data\\lab1\\ShahzadpurHH.tif")
    img2 = cv2.imread("C:\\Users\\ma\\project\\Scripts\\data\\lab1\\ShahzadpurHH.tif")
    (rows, cols) = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
    res = cv2.warpAffine(img, M, (cols, rows))
    res2 = cv2.warpAffine(img2, M, (cols, rows))
    cv2.imwrite('result.tiff', res)
    cv2.imwrite('result.tiff', res2)
    plt.imshow(res)
    plt.imshow(res2)
    plt.show()
except IOError:
    print('Error while reading files !!!')"""

"reducing tiff file size"
"""try:
    # Read image from disk.
    img = cv2.imread("C:\\Users\\ma\\project\\Scripts\\data\\lab1\\ShahzadpurHH.tif")
    img2 = cv2.imread("C:\\Users\\ma\\project\\Scripts\\data\\lab1\\ShahzadpurHV.tif")
    (height, width) = img.shape[:2]
    res = cv2.resize(img, (int(width / 2), int(height / 2)), interpolation=cv2.INTER_CUBIC)
    res2 = cv2.resize(img2, (int(width / 2), int(height / 2)), interpolation=cv2.INTER_CUBIC)

    cv2.imwrite('result.tiff', res)
    cv2.imwrite('result.tiff', res2)
    plt.imshow(res)
    plt.imshow(res2)
    plt.show()
except IOError:
    print('Error while reading files !!!')"""

'image statistics'
"""img = Image.open("C:\\Users\\ma\\project\\Scripts\\data\\lab1\\ShahzadpurHH.tif")
img2 = Image.open("C:\\Users\\ma\\project\\Scripts\\data\\lab1\\ShahzadpurHV.tif")
img_array = np.array(img)
mean = np.mean(img_array)
dev= np.std(img_array)
min = np.min(img_array)
max = np.max(img_array)
img2_array = np.array(img2)
mean2 = np.mean(img2_array)
dev2= np.std(img2_array)
min2 = np.min(img2_array)
max2 = np.max(img2_array)
print('minimum value of HH image = ',min ,"\n","maximum value of HH image =", max,"\n",'meanof HH image =', mean,"\n","standard dev of HH image=",dev )
print('minimum value of HV image  = ',min2 ,"\n","maximum value of HV image =", max2,"\n",'mean of HV image =', mean2,"\n","standard dev of HV image=",dev2 )"""

'open a tiff file'
"""img = Image.open("C:\\Users\\ma\\project\\Scripts\\data\\lab1\\ShahzadpurHH.tif")
img2 = Image.open("C:\\Users\\ma\\project\\Scripts\\data\\lab1\\ShahzadpurHV.tif")
show(img)
show(img2)
pixels = img.load()
pixels2 = img2.load()"""

'second programme Meta data about tiff images'
"""a=rasterio.open("C:\\Users\\ma\\project\\Scripts\\data\\lab1\\ShahzadpurHH.tif")
b=rasterio.open("C:\\Users\\ma\\project\\Scripts\\data\\lab1\\ShahzadpurHV.tif")
img = Image.open("C:\\Users\\ma\\project\\Scripts\\data\\lab1\\ShahzadpurHH.tif")
img2 = Image.open("C:\\Users\\ma\\project\\Scripts\\data\\lab1\\ShahzadpurHV.tif")
width= a.width
height=a.height
width2= b.width
height2=b.height
num_bands = a.count
num_bands2 = b.count
print("Image(HH) Size  (Width x Height):", width, "x", height)
print("Image (HV) Size (Width x Height):", width2, "x", height2)
print("Number of Bands of HH:", num_bands)
print("Number of Bands of HH:", num_bands2)
plt.imshow(img)
plt.imshow(img2)
plt.show()"""










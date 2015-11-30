import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt

img = cv2.imread(sys.argv[1])

print img.shape
rows, cols, d = img.shape
crow, ccol = rows/2 , cols/2
N = 4

dct_arr = np.zeros((rows,cols,3), np.uint8)
a = []
for i in xrange(3):
    imf = np.float32(img[:,:,i]) /255.0  # float conversion/scale
    dst = cv2.dct(imf)# the dct
    msk = np.zeros((rows, cols), np.uint8)
    msk[:N,:N] = 1
    dst = msk * dst
    print "Mask: ", dst[:N, :N]
    tmp = cv2.idct(dst);
    cv2.normalize(tmp,tmp,0,255,cv2.NORM_MINMAX)
    img2 = np.uint8(tmp)
    print tmp[299, 199], img2[299, 199]
    a.append(img2)
dct_arr = (np.dstack((a[0],a[1],a[2])) )

print "Byte spent: ", N * N * 3 * 4

plt.subplot(121)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(122)
plt.imshow(cv2.cvtColor(dct_arr, cv2.COLOR_BGR2RGB))
plt.show()

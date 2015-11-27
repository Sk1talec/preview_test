import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('input3.png')

print img.shape
rows, cols, d = img.shape
crow, ccol = rows/2 , cols/2
N = 2

"""
dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# create a mask first, center square is 1, remaining all zeros
mask = np.zeros((rows,cols,2),np.uint8)
mask[crow-N:crow+N, ccol-N:ccol+N] = 1

# apply mask and inverse DFT
fshift = dft_shift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
"""
dct_arr = np.zeros((rows,cols,3), np.uint8)
a = []
for i in xrange(3):
    imf = np.float32(img[:,:,i]) /255.0  # float conversion/scale
    dst = cv2.dct(imf)# the dct
    print "val" , dst[0,0]
    msk = np.zeros((rows, cols), np.uint8)
    msk[:N,:N] = 1
    dst = msk * dst
    img2 = cv2.idct(dst)
    a.append(img2)
dct_arr = (np.dstack((a[0],a[1],a[2])) * 255.999) .astype(np.uint8)

print "Byte spent: ", N * N * 3 * 4

plt.subplot(121)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(122)
plt.imshow(cv2.cvtColor(dct_arr, cv2.COLOR_BGR2RGB))
plt.show()

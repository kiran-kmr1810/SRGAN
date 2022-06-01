import math
import numpy as np
import cv2
import glob
import os.path as osp
import matplotlib.pyplot as plt

path_to_GT = "GT/*"


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if (mse == 0):
        return 100
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5] 
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    ssim_out = 1-ssim_map.mean()
    return ssim_out


def calc_ssim(img1, img2):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def gt_finder(idx):
    idx1 = -1
    for path in glob.glob(path_to_GT):
        idx1 += 1
        if idx1 == idx:
            return np.array(cv2.imread(path))


path_to_ESRGAN = 'ESRGAN/*'

psnre = []
ssime = []

idx = 0
print("Calculating for ESRGAN")
for path in glob.glob(path_to_ESRGAN):
    idx += 1
    try:
        base = osp.splitext(osp.basename(path))[0]
        image1 = np.array(cv2.imread(path))
        image2 = gt_finder(idx)
        image1.resize(image2.shape)
        psnre.append((PSNR(image1, image2))*1000)
        ssime.append((calc_ssim(image1, image2)))
    except:
        continue
print(psnre)
print(ssime)

path_to_OURMODEL = 'Our_model/*'

psnro = []
ssimo = []

print("Calculating for Our_model")
idx = 0
for path in glob.glob(path_to_OURMODEL):
    idx += 1
    try:
        base = osp.splitext(osp.basename(path))[0]
        image1 = np.array(cv2.imread(path))
        image2 = gt_finder(idx)
        image1.resize(image2.shape)
        psnro.append((PSNR(image1, image2))*1000)
        ssimo.append((calc_ssim(image1, image2)))
    except:
        continue

print(psnro)
print(ssimo)

path_to_SRGAN = 'SRGAN/*'

psnrs = []
ssims = []

print("Calculating for SRGAN")
idx = 0
for path in glob.glob(path_to_SRGAN):
    idx += 1
    try:
        base = osp.splitext(osp.basename(path))[0]
        image1 = np.array(cv2.imread(path))
        image2 = gt_finder(idx)
        image1.resize(image2.shape)
        psnrs.append((PSNR(image1, image2))*1000)
        ssims.append((calc_ssim(image1, image2)))
    except:
        continue

print(psnrs)
print(ssims)

path_to_g = 'Ground/*'
print("Calculating for Ground Truth")

psnrg = []
ssimg = []
idx = 0
for path in glob.glob(path_to_g):
    idx += 1
    try:
        base = osp.splitext(osp.basename(path))[0]
        image1 = np.array(cv2.imread(path))
        image2 = gt_finder(idx)
        image1.resize(image2.shape)
        psnrg.append((PSNR(image1, image2))*1000)
        ssimg.append((calc_ssim(image1, image2)))
    except:
        continue

print(psnrg)
print(ssimg)

X = ['1','2','3','4','5','6','7','8','9']

X_axis = np.arange(len(X))


plt.bar(X_axis - 0.2 , psnrs , 0.2 , label = 'SRGAN')
plt.bar(X_axis  , psnre , 0.2 , label = 'ESRGAN')
plt.bar(X_axis + 0.2 , psnro , 0.2 , label = 'Our Model')
  
plt.xticks(X_axis, X)
plt.xlabel("Each Test Image")
plt.ylabel("PSNR Value")
plt.legend()
plt.show()

X = ['1','2','3','4','5','6','7','8','9']

X_axis = np.arange(len(X))


plt.bar(X_axis - 0.2 , ssims , 0.2 , label = 'SRGAN')
plt.bar(X_axis  , ssime , 0.2 , label = 'ESRGAN')
plt.bar(X_axis + 0.2 , ssimo , 0.2 , label = 'Our Model')

plt.xticks(X_axis, X)
plt.xlabel("Each Test Image")
plt.ylabel("SSIM Value")
plt.legend()
plt.show()

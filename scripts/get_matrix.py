#!/usr/bin/env python3

import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt
#from skimage.filters import gaussian
import pandas as pd
from scipy.ndimage import zoom
from skimage.feature import canny,blob_doh
from skimage.filters import threshold_otsu,hessian
from skimage.draw import disk

import gc

# =========================
# USER INPUTS
# =========================

MARKER_FILES = {
    "CD10": "/media/Lawrenson_Lab_NAS/uthscsa/collab_data/courtois_cellDive/SL260089-CD26038_S22-70591-B1-BEME-342-4-US/raw/CD26038_1.0.4_R000_Cy7_CD10-CF750_FINAL_AFR_F.ome.tif",
    "KRT8": "/media/Lawrenson_Lab_NAS/uthscsa/collab_data/courtois_cellDive/SL260089-CD26038_S22-70591-B1-BEME-342-4-US/raw/CD26038_2.0.4_R000_Cy5_KRT8-18-AF647_FINAL_AFR_F.ome.tif",
    "PGP95": "/media/Lawrenson_Lab_NAS/uthscsa/collab_data/courtois_cellDive/SL260089-CD26038_S22-70591-B1-BEME-342-4-US/raw/CD26038_1.0.4_R000_Cy3_PGP9-5-AF555_FINAL_AFR_F.ome.tif",
    "CD45": "/media/Lawrenson_Lab_NAS/uthscsa/collab_data/courtois_cellDive/SL260089-CD26038_S22-70591-B1-BEME-342-4-US/raw/CD26038_1.0.4_R000_Cy5_CD45-AF647_FINAL_AFR_F.ome.tif",
    #"DAPI_R1": "/media/Lawrenson_Lab_NAS/uthscsa/collab_data/courtois_cellDive/SL260089-CD26038_S22-70591-B1-BEME-342-4-US/raw/CD26038_1.0.4_R000_DAPI__FINAL_F.ome.tif",
    #"DAPI_R2": "/media/Lawrenson_Lab_NAS/uthscsa/collab_data/courtois_cellDive/SL260089-CD26038_S22-70591-B1-BEME-342-4-US/raw/CD26038_2.0.4_R000_DAPI__FINAL_F.ome.tif",
    "CD20": "/media/Lawrenson_Lab_NAS/uthscsa/collab_data/courtois_cellDive/SL260089-CD26038_S22-70591-B1-BEME-342-4-US/raw/CD26038_2.0.4_R000_FITC_CD20-AF488_FINAL_AFR_F.ome.tif",
}
AUTOFLUORESCENCE_FILE = "/media/Lawrenson_Lab_NAS/uthscsa/collab_data/courtois_cellDive/SL260089-CD26038_S22-70591-B1-BEME-342-4-US/raw/CD26038_1.0.1_R000_DAPI_AF_F.ome.tif"

DAPI_MARKERS = ["DAPI_R1", "DAPI_R2"]


BIN_SIZE = 40   # at 50 KRT8 fragments
SIGMA=3
OUTPUT_DIR = "/media/Lawrenson_Lab_NAS/uthscsa/group_data/CosMx_temp/SL260089/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# 1. FUNCTIONS
# =========================
def load_tif(path):

    img = tifffile.imread(path)

    if img.ndim > 2:
        img = img.squeeze()

    return img.astype(np.float32)


def bin_image(img, bin_size):
    H, W = img.shape
    H_trim = (H // bin_size) 
    W_trim = (W // bin_size)
    img = img[:H_trim* bin_size, :W_trim* bin_size]

    binned = img.reshape(
        H_trim,
        bin_size,
        W_trim,
        bin_size).mean(axis=(1,3))
    return binned


def compute_alpha(marker_vec, af_vec):#SLOW
    pos_mask=(marker_vec>0)
    X = af_vec[pos_mask].reshape(-1,1)
    y = marker_vec[pos_mask]
    model = HuberRegressor()
    model.fit(X, y)
    alpha = model.coef_[0]
    # conservative cap
    alpha = np.clip(alpha, 0.01, 0.5)
    return alpha

# =========================
# 2. AF
# =========================
print("Reading AF file")
af_img = load_tif(AUTOFLUORESCENCE_FILE)
af_img=np.arcsinh(af_img/ 5)
#small = af_img[::8, ::8]
#plt.figure(figsize=(8,8))
#plt.imshow(af_img, cmap='inferno',
#           vmin=np.percentile(af_img, 5),
#           vmax=np.percentile(af_img, 99))
#plt.colorbar()
#plt.title("AF")

#plt.savefig(os.path.join(OUTPUT_DIR, "AF.png"))    

# =========================
# 3. MASK
# =========================
print("Building tissue mask...")
thr_af=np.percentile(af_img, 75)
#thr_dapi=np.percentile(dapi_avg, 75)
#print("AF threshold:", thr_af)
#print("DAPI threshold:", thr_dapi)
pixel_mask = (af_img > thr_af) 
#small = pixel_mask[::8, ::8]
#plt.figure(figsize=(8,8))
#plt.imshow(pixel_mask, cmap='inferno')
#plt.colorbar()
#plt.title("Mask")

print("Computing occupancy...")
occupancy = bin_image(pixel_mask,BIN_SIZE)
meta_mask = occupancy > .8# with dapi mask was .4 

neg_mask =np.clip(1-meta_mask, 0,None)
plt.figure(figsize=(8,8))
plt.imshow(neg_mask, cmap=plt.cm.gray)
plt.colorbar()
plt.title("Mask")
#plt.savefig(os.path.join(OUTPUT_DIR, "Neg_mask.png"))

# bin mask + get coordinates
print("Binning AF...")
af_bin = bin_image(af_img,BIN_SIZE)
#dapi_bin = bin_image(dapi_avg,BIN_SIZE)

mask_flat = meta_mask.reshape(-1)
af_flat = af_bin.reshape(-1)[mask_flat]
#dapi_flat = dapi_bin.reshape(-1)[mask_flat]

# ============================================================
# 4. MARKERS
# ============================================================
marker_names = [m for m in MARKER_FILES]

print("Processing markers...")

results = []

coords_y, coords_x = np.where(meta_mask)

for marker in marker_names:
    print(f"Loading {marker}")
    img = load_tif(MARKER_FILES[marker])
    #img = img[::8, ::8]
    #im1=ax1.imshow(img, cmap='inferno',
    #           vmin=np.percentile(img, 5),
    #           vmax=np.percentile(img, 99))
    thres=threshold_otsu(img)
    img_corr=img.copy()
    img_corr[img_corr<thres]=0

    if(np.count_nonzero(img_corr)>10000000000):
        print("bad otsu")
        thres=np.percentile(img,0.99)
        img_corr[img_corr<thres]=0


    edges = canny(img_corr,low_threshold=.75,
                  high_threshold=.99,
                  use_quantiles=True,
                  mode='reflect',
                  sigma=SIGMA,
                  mask=pixel_mask)
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10, 8))
    small = edges[::8, ::8]
    small = np.clip(1-small, 0,None)
    im1=ax1.imshow(small, cmap="binary")

    img_corr[edges==0] = 0
    img_corr = np.arcsinh(img_corr/ 5)    
    img_bin = bin_image(img_corr,BIN_SIZE)
    im2=ax2.imshow(img_bin, cmap=plt.cm.gray)
    plt.tight_layout() # Adjusts spacing to prevent overlap
    plt.show()

    if(marker == "PGP95"):
        img_shapes=hessian(img_bin,alpha=.1)#black_ridges=False?
        img_bin[img_shapes==0] = 0

    if("CD45" in ["CD45","CD20"]):
        img_shapes= blob_doh(img_bin,max_sigma=30)
        blob_mask = np.zeros(img_bin.shape[:2], dtype=bool)
        for y, x, sigma in img_shapes:
            radius = sigma * np.sqrt(2)  # Standard scaling for DoH/LoG blobs
            rr, cc = disk((y, x), radius, shape=blob_mask.shape)
            blob_mask[rr, cc] = True
        img_bin[blob_mask==0]=0

    
    thres=threshold_otsu(img_bin)
    img_bin[img_bin<thres]=0
    marker_vec = img_bin.reshape(-1)[mask_flat]
    results.append(marker_vec)

    gc.collect()
# ============================================================
# 5. SAVE FINAL MATRIX
# ============================================================
pixel_matrix = np.stack(results, axis=1)

df = pd.DataFrame(pixel_matrix, columns=marker_names)
df["x"] = coords_x
df["y"] = coords_y
df["AF"]=af_flat
df = df[["x", "y","AF"] + marker_names]

out_csv = os.path.join(
    OUTPUT_DIR,
    "meta_pixel_matrix.csv"
)

df.to_csv(
    out_csv,
    index=False
)

print("Done.")
print(df.shape)
print(f"Saved: {out_csv}")

#plt.figure(figsize=(6,6))
#plt.scatter(df["x"], -df["y"], c=df["CD45"], cmap="viridis", s=1)
#plt.show()

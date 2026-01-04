# =========================
# IEEE IMAGE FEATURE PIPELINE
# =========================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import shannon_entropy
import os

# -------------------------
# CONFIGURATION
# -------------------------
IMAGE_PATH = "D:\sudo-solo\dataset\dusty\dusty_19.jpg"  # CHANGE THIS
OUTPUT_DIR = "ieee_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# LOAD IMAGE
# -------------------------
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise ValueError("Image not found. Check IMAGE_PATH.")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# -------------------------
# RGB HISTOGRAM
# -------------------------
plt.figure(figsize=(8,4))
colors = ('r','g','b')
for i, col in enumerate(colors):
    hist = cv2.calcHist([img_rgb],[i],None,[256],[0,256])
    plt.plot(hist, color=col, linewidth=1.5)

plt.title("RGB Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/rgb_histogram.png", dpi=300)
plt.close()

# -------------------------
# GRAYSCALE HISTOGRAM
# -------------------------
plt.figure(figsize=(8,4))
hist_gray = cv2.calcHist([gray],[0],None,[256],[0,256])
plt.plot(hist_gray, color='black', linewidth=1.5)
plt.title("Grayscale Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/gray_histogram.png", dpi=300)
plt.close()

# -------------------------
# LUMINANCE CALCULATION
# -------------------------
luminance = (
    0.299 * img_rgb[:,:,0] +
    0.587 * img_rgb[:,:,1] +
    0.114 * img_rgb[:,:,2]
)

plt.figure(figsize=(6,6))
plt.imshow(luminance, cmap='gray')
plt.title("Luminance Image")
plt.axis('off')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/luminance.png", dpi=300)
plt.close()

# -------------------------
# FEATURE EXTRACTION
# -------------------------
features = {}

# Statistical Features
features["Mean Intensity"] = np.mean(gray)
features["Standard Deviation"] = np.std(gray)
features["Entropy"] = shannon_entropy(gray)
features["Mean Luminance"] = np.mean(luminance)

# -------------------------
# GLCM TEXTURE FEATURES
# -------------------------
glcm = graycomatrix(
    gray,
    distances=[1],
    angles=[0],
    levels=256,
    symmetric=True,
    normed=True
)

features["GLCM Contrast"] = graycoprops(glcm, 'contrast')[0,0]
features["GLCM Energy"] = graycoprops(glcm, 'energy')[0,0]
features["GLCM Homogeneity"] = graycoprops(glcm, 'homogeneity')[0,0]

# -------------------------
# EDGE DETECTION & DENSITY
# -------------------------
edges = cv2.Canny(gray, 100, 200)
features["Edge Density"] = np.sum(edges > 0) / edges.size

plt.figure(figsize=(6,6))
plt.imshow(edges, cmap='gray')
plt.title("Canny Edge Detection")
plt.axis('off')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/edges.png", dpi=300)
plt.close()

# -------------------------
# LOCAL BINARY PATTERN
# -------------------------
radius = 1
points = 8 * radius
lbp = local_binary_pattern(gray, points, radius, method="uniform")

features["LBP Mean"] = np.mean(lbp)
features["LBP Std"] = np.std(lbp)

# -------------------------
# SHARPNESS MEASURE
# -------------------------
features["Laplacian Variance"] = cv2.Laplacian(gray, cv2.CV_64F).var()

# -------------------------
# SAVE FEATURE TABLE
# -------------------------
df = pd.DataFrame.from_dict(features, orient="index", columns=["Value"])
df.index.name = "Feature"

df.to_csv(f"{OUTPUT_DIR}/image_features.csv")
df.to_excel(f"{OUTPUT_DIR}/image_features.xlsx")

# -------------------------
# PRINT RESULTS
# -------------------------
print("\n===== EXTRACTED IMAGE FEATURES =====\n")
print(df)

print("\nAll outputs saved in folder:", OUTPUT_DIR)

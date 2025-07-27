import cv2
import numpy as np
import matplotlib.pyplot as plt
import keras
from tensorflow.keras.models import load_model
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# === Configuration ===
IMG_W, IMG_H = 200, 50  # Adjust if your models use different size
CHARS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
MAX_LENGTH = 5  # Adjust if your CAPTCHAs have different length

# === Enable Lambda deserialization ===
keras.config.enable_unsafe_deserialization()

# === Load Models ===
generator = load_model("Model/adv_refiner.keras", compile=False)
solver    = load_model("Model/captcha_model.keras", compile=False)

# === Select a CAPTCHA image ===
Tk().withdraw()  # Hide main window
fname = askopenfilename(title="Select CAPTCHA image", filetypes=[("Image files", "*.png *.jpg *.jpeg")])
orig = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)

# === Preprocess image ===
img = cv2.resize(orig, (IMG_W, IMG_H)).astype(np.float32) / 255.0
x = img[np.newaxis, ..., np.newaxis]  # Shape: (1, H, W, 1)

# === Solver prediction on original image ===
preds_orig = solver.predict(x, verbose=0)
pred_orig = "".join(CHARS[np.argmax(preds_orig[i][0])] for i in range(MAX_LENGTH))

# === Refine image ===
refined = generator.predict(x)[0, :, :, 0]
x_ref = refined[np.newaxis, ..., np.newaxis]

# === Solver prediction on refined image ===
preds_ref = solver.predict(x_ref, verbose=0)
pred_ref = "".join(CHARS[np.argmax(preds_ref[i][0])] for i in range(MAX_LENGTH))

# === Display results ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
ax1.imshow(orig, cmap='gray')
ax1.set_title(f"Original\nSolver: {pred_orig}")
ax1.axis('off')

ax2.imshow((refined * 255).astype(np.uint8), cmap='gray')
ax2.set_title(f"Refined\nSolver: {pred_ref}")
ax2.axis('off')

plt.tight_layout()
plt.show()
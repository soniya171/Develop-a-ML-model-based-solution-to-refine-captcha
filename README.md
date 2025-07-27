# 🧠 CAPTCHA Refiner using CNN & OpenCV

A smart ML-based tool to clean and decode distorted CAPTCHA images 🖼️ using image processing techniques 🧼 and a Convolutional Neural Network (CNN) 🤖.

---

## ✨ Project Highlights

🔹 Clean & preprocess noisy CAPTCHA images  
🔹 Segment characters using contour detection  
🔹 Recognize characters using a trained CNN  
🔹 Improve decoding accuracy for complex CAPTCHAs  
🔹 Simple, modular, and ready-to-use

---

## ⚙️ Tech Stack

- 🐍 Python  
- 📷 OpenCV  
- 🔢 NumPy, Matplotlib  
- 🤖 TensorFlow / Keras (CNN Model)  
- 📁 Jupyter Notebooks for training & EDA

---

## 📂 Directory Structure

captcha-refiner/
├── images/ # Raw CAPTCHA samples
├── results/ # Cleaned & predicted outputs
├── model/ # Trained CNN model (.h5)
├── refine_captcha.py # Preprocessing & segmentation
├── cnn_predict.py # CNN prediction logic
├── train_model.ipynb # CNN model training notebook
├── utils.py # Helper functions
├── README.md
└── requirements.txt

---

## 🚀 Getting Started

### 🔧 Install Dependencies

```bash
git clone https://github.com/yourusername/captcha-refiner.git
cd captcha-refiner
pip install -r requirements.txt

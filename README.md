# 🦴 Pediatric Bone Age Detection AI 🏥

## 📌 Overview
This is an **open-source AI model** for **predicting bone age from pediatric X-ray images**, based on the **RSNA Bone Age dataset**. The model uses **DeepLabV3+** for segmentation and **EfficientNetV2-M** for bone age regression. It is optimized with techniques like test-time augmentation (TTA).

The model is deployed via a **FastAPI backend** and is accessible through an API, which is integrated with a **React web app**.

---

## ⚙️ Features
✅ **Bone age prediction from X-ray images**  
✅ **Automated segmentation with DeepLabV3+**  
✅ **EfficientNet-V2M for analysis**  
✅ **Test-time augmentation for better accuracy**  
✅ **Supports inference via API (FastAPI-based)**  
✅ **Optimized for Google Colab (uses A100 GPU)**  
✅ **Fully open-source under OpenRAIL License**  

---

## 🏗 Model Architecture
### **1️⃣ Image Segmentation**
- Model: **DeepLabV3+ (ResNet-50 backbone)**
- Purpose: Extracts the region of interest (bones)

### **2️⃣ Bone Age Regression**
- Model: **EfficientNetV2-M**
- Additional Features: **Gender input as auxiliary feature**
- Training:
  - **10,000 training images** (RSNA dataset)
  - **1,200 validation images**
  - **Augmentations**

### **3️⃣ Calibration Model**
- **Linear regression layer** for final calibration

---

## 🚀 API Usage
This model is deployed via a **FastAPI backend** on Hugging Face Spaces. 

### **Health Check**
```bash
curl -X GET "https://ameyakawthalkar-boneagealpha.hf.space/health"

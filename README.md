
# 🧠 Brain Tumor MRI Image Classification

This project leverages **deep learning** and **transfer learning** to classify brain MRI images into four categories: **glioma**, **meningioma**, **pituitary tumor**, and **no tumor**. It achieves high performance through a fine-tuned **ResNet18 model**, robust data augmentation, and is deployed via an interactive **Streamlit web app**.

---

## 📌 Problem Statement

Early and accurate detection of brain tumors is crucial for effective treatment. This project aims to assist radiologists by providing an AI-based tool that classifies MRI brain scans by tumor type in real-time.

---

## 🧠 Model Highlights

- 📚 **Model**: ResNet18 pretrained on ImageNet
- 🔄 **Fine-Tuning**: Final and base layers fine-tuned for medical domain
- 🎯 **Test Accuracy**: **95.53%**
- 📈 **F1 Score**: **96%**
- 💾 Trained on 2443 labeled MRI images

---

## 📁 Dataset

- **Classes**: Glioma, Meningioma, Pituitary, No Tumor
- **Total Images**: 2443
- **Split**:
  - Train: 1695
  - Validation: 502
  - Test: 246

---

## 🚀 Features

- ✅ Transfer learning using ResNet18
- ✅ Robust data augmentation (rotation, flips, brightness, zoom)
- ✅ High generalization performance
- ✅ Streamlit deployment for real-time predictions
- ✅ Clean, modular PyTorch code

---

## 📊 Evaluation Results

| Class       | Precision | Recall | F1-score |
|-------------|-----------|--------|----------|
| Glioma      | 0.98      | 0.99   | 0.99     |
| Meningioma  | 0.95      | 0.90   | 0.93     |
| No Tumor    | 0.95      | 0.94   | 0.94     |
| Pituitary   | 0.94      | 0.98   | 0.96     |

**Test Accuracy**: **95.53%**  
** Avg F1-score**: **96%**

---

## 🖼 Streamlit App Demo

> Upload an MRI image and instantly get the predicted tumor type with model confidence.

### 🔧 Run the app locally:
```bash
streamlit run app.py
```

## 👩‍⚕️ Use Cases

- **Clinical Decision Support**: Assist radiologists with quick second opinions
- **Telemedicine**: Enable MRI diagnosis in low-resource areas
- **Research & Trials**: Stratify patient data by tumor class

---
## Files

- **Tumour0.pynb**: model development
- **Tumour1.py**: streamlit file

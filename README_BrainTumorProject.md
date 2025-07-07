
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
- 📈 **Macro F1 Score**: **96%**
- 💾 Trained on 2443 labeled MRI images

---

## 📁 Dataset

- **Source**: [Roboflow: Labeled MRI Brain Tumor Dataset](https://universe.roboflow.com/ali-rostami/labeled-mri-brain-tumor-dataset)
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
**Macro Avg F1-score**: **96%**

---

## 🖼 Streamlit App Demo

> Upload an MRI image and instantly get the predicted tumor type with model confidence.

### 🔧 Run the app locally:
```bash
streamlit run app.py
```

### 🖼 Sample Output:
- Tumor Type: `Pituitary`
- Confidence: `97.45%`

---

## 🧪 Project Structure

```
brain-tumor-classifier/
├── app.py                   ← Streamlit app
├── models/
│   └── resnet18_brain_tumor_final.pth
├── scripts/
│   ├── train_custom_cnn.py
│   └── train_resnet18.py
├── utils.py                 ← Preprocessing, model loader (optional)
├── requirements.txt
└── README.md
```

---

## 💻 Installation

```bash
git clone https://github.com/your-username/brain-tumor-classifier.git
cd brain-tumor-classifier
pip install -r requirements.txt
```

---

## 📦 Requirements

- Python 3.8+
- PyTorch
- torchvision
- streamlit
- matplotlib
- seaborn
- scikit-learn
- Pillow

---

## 📌 Future Work

- Add Grad-CAM for visual tumor localization
- Integrate Flask or FastAPI for API deployment
- Train on larger, multi-modal datasets (e.g. DICOM series)

---

## 👩‍⚕️ Use Cases

- **Clinical Decision Support**: Assist radiologists with quick second opinions
- **Telemedicine**: Enable MRI diagnosis in low-resource areas
- **Research & Trials**: Stratify patient data by tumor class

---

## 👨‍💻 Author

**Your Name**  
_Data Science & Deep Learning Enthusiast_  
📧 your.email@example.com

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).

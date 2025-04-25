# 🧠 Deep Learning Project: Pneumonia Detection System

This is a **mini-project** that uses **deep learning** techniques to build a **Pneumonia Detection System** from chest X-ray images. The model classifies whether a patient has pneumonia or not, using **Convolutional Neural Networks (CNN)** and **Transfer Learning**.

---

## 🩺 Project Description

The goal of this project is to automate the detection of pneumonia using medical imaging. We used a public dataset of chest X-rays and implemented various CNN architectures including **VGG16**, **ResNet50**, and a **custom CNN model** to identify patterns associated with pneumonia.

> 📂 Dataset: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) from Kaggle

---

## 🔍 Key Features

- 🖼️ Preprocessing and augmentation of chest X-ray images
- 🧠 CNN architecture built from scratch
- 🧬 Transfer Learning with VGG16 & ResNet50
- 📈 Evaluation using accuracy, precision, recall, F1-score
- 📊 Visualization of training and prediction results
- 📦 Model saved for later use

---

## 🧪 Tech Stack

- **Language**: Python
- **Libraries**: TensorFlow, Keras, OpenCV, Matplotlib, NumPy, scikit-learn
- **Models**: Custom CNN, VGG16, ResNet50
- **Tools**: Jupyter Notebook, Google Colab

---

## 📁 Project Structure

```
Pneumonia-Detection-CNN/
├── dataset/                     # Chest X-ray image dataset (train/test/val)
├── notebooks/
│   ├── 1_data_preprocessing.ipynb
│   ├── 2_custom_cnn_model.ipynb
│   ├── 3_transfer_learning_vgg16.ipynb
│   └── 4_resnet_model.ipynb
├── saved_models/                # Trained model weights
├── plots/                       # Training accuracy/loss graphs
├── README.md
├── requirements.txt
```

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/Pneumonia-Detection-CNN.git
cd Pneumonia-Detection-CNN
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebooks

Use Jupyter Notebook or Google Colab to open the notebooks inside the `notebooks/` folder.

---

## 📊 Results

| Model         | Accuracy | Precision | Recall | F1-Score |
|---------------|----------|-----------|--------|----------|
| Custom CNN    | 84.7%    | 85.2%     | 84.0%  | 84.6%    |
| VGG16         | 91.3%    | 91.0%     | 92.1%  | 91.5%    |
| ResNet50      | 93.2%    | 93.5%     | 92.8%  | 93.1%    |

---

## 🖼 Sample Predictions

<img src="plots/sample_predictions.png" width="60%" alt="Model Predictions" />

---

## ✅ Key Learnings

- Deep understanding of CNN architecture and layers
- Transfer learning improves performance on small datasets
- Image preprocessing and augmentation are crucial for generalization
- Evaluation metrics beyond accuracy give better model insights

---

## 📧 Contact

👤 Name: Ria Kalra  
📩 Email: [29861it@gmail.com](mailto:youremail@example.com)  
🔗 LinkedIn: [https://www.linkedin.com/in/ria-kalra-604788230/](https://linkedin.com/in/yourprofile)

---

## 📜 License

This project is licensed under the MIT License 

---

## 🙏 Acknowledgements

- Kaggle for the dataset  
- TensorFlow/Keras documentation  
- Medical professionals contributing to open-source datasets
```


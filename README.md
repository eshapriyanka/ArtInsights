- Designed a two-part computer vision pipeline to (1) apply artistic styles to input images, and (2) detect whether an artwork is AI-generated or human-made.
- Implemented VGG-based neural style transfer and a custom CNN for stylization, achieving high visual fidelity on Cubism-styled renderings.
- Built and benchmarked fake detection models using DINOv2 (Champion) and EfficientNet (Challenger), evaluating performance across pure and hybrid datasets with ROC-AUC and F1 metrics.
- Deployed top-performing models into a real-time Streamlit web app for live testing, enabling users to apply style transfer or detect image authenticity through an interactive interface.

- Applied best practices in PyTorch, including label encoding, normalization, early stopping, learning rate scheduling, and custom loss functions.

🎨 ArtInsights: Neural Style Transfer & AI Artwork Detection

ArtInsights is a Computer Vision application that combines **Neural Style Transfer** and **AI-Generated Artwork Detection** into a single interactive platform. The project enables users to transform images into artistic styles and analyze whether an artwork is human-created or AI-generated.

Built using **TensorFlow**, **DINOv2**, **Scikit-learn**, and **Streamlit**, the system demonstrates the integration of deep learning, feature extraction, image classification, and web deployment.

---

## 📌 Overview

The project consists of two major modules:

### 🎨 Neural Style Transfer

Transforms uploaded images into artistic renderings while preserving the original content and structure.

### 🔍 AI Artwork Detection

Analyzes uploaded artwork images and predicts whether they are human-created or AI-generated using DINOv2 feature embeddings and a trained SVM classifier.

---

## ✨ Features

* 🎨 Neural Style Transfer
* 🔍 AI Artwork Detection
* 🧠 DINOv2 Feature Extraction
* 📊 Confidence Score Visualization
* 🌐 Interactive Streamlit Interface
* 📁 Drag-and-Drop Image Upload
* ⚡ Real-Time Predictions
* 📈 Machine Learning-Based Classification

---

## 🏗️ System Architecture

```text
                    User Upload
                          │
            ┌─────────────┴─────────────┐
            │                           │
            ▼                           ▼

   Neural Style Transfer        Artwork Authentication

            │                           │
            ▼                           ▼

  TensorFlow Style Model     DINOv2 Feature Extraction

            │                           │
            ▼                           ▼

     Stylized Artwork         SVM Classification

            │                           │
            ▼                           ▼

        Display Result      Human / AI Prediction
```

---

## 🧠 Technologies Used

| Technology   | Purpose                   |
| ------------ | ------------------------- |
| Python       | Core Programming Language |
| TensorFlow   | Neural Style Transfer     |
| DINOv2       | Feature Extraction        |
| Scikit-learn | Artwork Classification    |
| Streamlit    | Interactive Web Interface |
| NumPy        | Numerical Computing       |
| Pillow (PIL) | Image Processing          |
| Joblib       | Model Serialization       |

---

## Running the Application

Launch the Streamlit application:

```bash
streamlit run app.py
```

Open your browser and navigate to:

```text
http://localhost:8501
```

---

## 🎨 Neural Style Transfer Workflow

1. Upload an image.
2. The image is preprocessed and converted into a TensorFlow tensor.
3. The style transfer model applies artistic transformations.
4. A stylized image is generated and displayed.

---

## 🔍 AI Artwork Detection Workflow

1. Upload an artwork image.
2. DINOv2 extracts high-level visual features.
3. Features are passed to a trained SVM classifier.
4. The model predicts:

   * Human-made Artwork
   * AI-generated Artwork
5. Confidence scores are displayed.

---

## 📊 Results

The artwork authentication module effectively distinguishes between human-created and AI-generated artworks using DINOv2 feature embeddings and machine learning classification.

The style transfer module successfully generates artistic transformations while preserving the content and structure of the original image.

---

## 🎯 Learning Outcomes

This project demonstrates:

* Computer Vision
* Neural Style Transfer
* Foundation Models (DINOv2)
* Feature Extraction
* Image Classification
* Model Deployment
* Streamlit Application Development
* End-to-End AI System Design

---


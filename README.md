# Rice Grain Classification using CNN and Transfer Learning

This project classifies five types of rice grains — Arborio, Basmati, Ipsala, Jasmine and Karacadag — using Convolutional Neural Networks (CNN) with Transfer Learning. 
---

# Project Overview

Rice grain classification is crucial for quality control and packaging in the food industry. This model leverages a pre-trained CNN (e.g., MobileNetV2) and fine-tunes the final layers to adapt to a custom rice grain dataset. The result is a robust image classifier capable of distinguishing between visually similar grain types.

---

# Project Structure

---

# Environment Setup

You can run this project in a virtual environment.

# Step-by-step:


# 1. Clone the repository
git clone https://github.com/RohitUpadhye/Rice_grain.git
---
cd Rice_grain

# 2. Create virtual environment
python -m venv env
env\Scripts\activate

# 3. Install required packages
pip install -r requirements.txt

#  Dependencies
All dependencies are listed in requirements.txt. 

tensorflow
---
numpy
---
pillow
---
streamlit
---
scikit-learn
---
matplotlib
---


How to Run
1. Train the Model
python ricemodel.py

2. Evaluate the model
python evaluation.py

3. Launch the Web UI
streamlit run app.py

This will open a browser UI to upload rice grain images and classify them.

# Results
Test Accuracy: 96.57%

Classification Report:

Precision: 0.94 - 0.99

Recall: 0.94 - 1.00

F1-Score: Avg 0.97

Confusion Matrix:
---
[[2215  0   18    1   16]
---
 [   7 2139    8   96    0]
 ---
 [   0    0 2250    0    0]
 ---
 [  64   21   56 2108    1]
 ---
 [  63    0   35    0 2152]]
---



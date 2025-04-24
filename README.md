# Rice Grain Classification using CNN and Transfer Learning

This project classifies five types of rice grains — Arborio, Basmati, Ipsala, Jasmine and Karacadag — using Convolutional Neural Networks (CNN) with Transfer Learning. 
---

# Project Overview

Rice grain classification is crucial for quality control and packaging in the food industry. This model leverages a pre-trained CNN (e.g., MobileNetV2) and fine-tunes the final layers to adapt to a custom rice grain dataset. The result is a robust image classifier capable of distinguishing between visually similar grain types.

---

# Project Structure

---

## ⚙️ Environment Setup

You can run this project in a virtual environment.

### Step-by-step:

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/rice-grain-classification.git
cd rice-grain-classification

# 2. Create virtual environment
python -m venv env
source env/bin/activate  # Windows: env\Scripts\activate

# 3. Install required packages
pip install -r requirements.txt



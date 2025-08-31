
#  House Price Prediction (Multimodal: Images + Tabular Data)

This project predicts **house prices in Southern California** using both **tabular features** (like number of bedrooms, bathrooms, square footage, etc.) and **house images**.
It leverages **deep learning (TensorFlow/Keras)** with a multimodal model that combines **CNN-based image features** and **MLP-based tabular features**.

---

##  Dataset

We use the **[SoCal House Prices and Images dataset](https://www.kaggle.com/datasets)** available on Kaggle.

* **socal.csv** → Contains metadata such as `price`, `beds`, `baths`, `sqft`, `zipcode`, etc.
* **images/** → Folder of house images (`1.jpg, 2.jpg, …`) corresponding to rows in the CSV.

---

##  Project Structure

```
├── socal.csv                # Tabular dataset
├── images/                  # Folder with house images
├── notebook.ipynb           # Jupyter Notebook with training pipeline
├── README.md                # Project description
└── requirements.txt         # Dependencies
```

---

##  Features

* Data preprocessing for **tabular** and **image** inputs
* **CNN (MobileNetV2 / EfficientNetB0)** for extracting image embeddings
* **Fully-connected layers** for processing tabular features
* **Fusion layer** that combines both modalities
* **Regression output** for predicting house prices

---

##  Setup & Installation

```bash
# Clone this repo
git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate    # (Linux/Mac)
venv\Scripts\activate       # (Windows)

# Install dependencies
pip install -r requirements.txt
```

Dependencies include:

* `tensorflow`
* `numpy`
* `pandas`
* `scikit-learn`
* `matplotlib`

---

##  Training the Model

Run the Jupyter Notebook:

```bash
jupyter notebook notebook.ipynb
```

Inside, you can:

* Load the dataset (`socal.csv` + images)
* Preprocess data
* Train the multimodal model
* Evaluate performance on Train/Validation/Test splits

---

##  Model Architecture

```
Tabular Input → Dense Layers ┐
                             ├─ Concatenate → Dense → Dense → Price
Image Input → CNN Backbone ┘
```

* **Backbone**: MobileNetV2 (default) or EfficientNetB0
* **Loss Function**: Mean Squared Error (MSE)
* **Optimizer**: Adam

---

## Results

* Train/Val/Test split: **70/15/15**
* Model learns to combine **structural features + visual features**
* Evaluation metrics: RMSE, MAE

---

##  Future Improvements

* Try larger CNN backbones (ResNet, EfficientNetV2)
* Hyperparameter tuning
* Data augmentation for images
* Incorporate location-based features

---

##  Author

Developed by **Anoosha**  – Computer Systems Engineering Student.


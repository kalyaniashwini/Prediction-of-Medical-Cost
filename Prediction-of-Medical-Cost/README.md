# Medical Cost Prediction

Predicts individual medical insurance costs using regression algorithms.

## Models Implemented
- Linear Regression
- Support Vector Regression (SVR)
- Decision Tree Regression
- Random Forest Regression

## Dataset
We use the [Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance).

**Option 1 – Include CSV**  
Place the dataset in `data/insurance.csv`.

**Option 2 – Download from Kaggle**
```bash
pip install kaggle
kaggle datasets download -d mirichoi0218/insurance
unzip insurance.zip -d data/

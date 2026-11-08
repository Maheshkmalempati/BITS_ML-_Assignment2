# Dry Bean Variety Classification – ML Assignment 2

**Your Name (20XXAAXXXXX)** – M.Tech (AIML), BITS Pilani WILP

- Live app: https://<your-app>.streamlit.app
- Repository: https://github.com/<username>/<repo>

## a. Problem statement
Dry beans of different varieties look similar, and sorting them by hand is slow and inconsistent. The goal is to predict the variety of a dry bean grain (one of 7 classes) from 16 shape features extracted from its image. Five classifiers - Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes and Support Vector Machine - are trained on the same data, compared on Accuracy, AUC, Precision, Recall, F1 and MCC, and served through an interactive Streamlit app.

## b. Dataset description
Source: UCI Machine Learning Repository - Dry Bean Dataset (ID 602), https://archive.ics.uci.edu/dataset/602/dry+bean+dataset. Images of 13,611 grains of 7 registered dry bean varieties were captured with a high-resolution camera, and 16 features (12 dimensional and 4 shape-form features) were extracted from each grain. After removing 68 duplicate rows, 13,543 instances were used. There are no missing values and all features are numeric. The data was split 80/20 (stratified, random_state=42): 10,834 training rows and 2,709 test rows (saved as test_data.csv). Features were median-imputed and standardised inside each model pipeline. Because this is a 7-class problem, AUC is one-vs-rest (weighted) and Precision/Recall/F1 are weighted averages.

**Class distribution (after de-duplication)**

| Class | Count |
|---|---|
| DERMASON | 3546 |
| SIRA | 2636 |
| SEKER | 2027 |
| HOROZ | 1860 |
| CALI | 1630 |
| BARBUNYA | 1322 |
| BOMBAY | 522 |

**Features**

| Feature | Description |
|---|---|
| Area | Area of the bean zone, in pixels |
| Perimeter | Length of the bean's border |
| MajorAxisLength | Length of the longest line that can be drawn through the bean |
| MinorAxisLength | Longest line perpendicular to the major axis |
| AspectRation | Ratio of major to minor axis length |
| Eccentricity | Eccentricity of the ellipse with the same moments as the region |
| ConvexArea | Pixels in the smallest convex polygon containing the bean |
| EquivDiameter | Diameter of a circle with the same area as the bean |
| Extent | Ratio of bean pixels to bounding-box pixels |
| Solidity | Ratio of bean pixels to convex-hull pixels |
| roundness | 4 * pi * Area / Perimeter^2 |
| Compactness | EquivDiameter / MajorAxisLength |
| ShapeFactor1 | Shape factor 1 |
| ShapeFactor2 | Shape factor 2 |
| ShapeFactor3 | Shape factor 3 |
| ShapeFactor4 | Shape factor 4 |

## c. GitHub repository link
https://github.com/<username>/<repo>

```
project-folder/
|-- app.py                     # Streamlit app
|-- ML_Assignment_2.ipynb      # training, evaluation and file generation
|-- requirements.txt
|-- README.md
|-- test_data.csv              # held-out test split used in the experiments
|-- data/Dry_Bean_Dataset.csv  # full dataset (CSV copy)
|-- .streamlit/config.toml     # app theme
|-- model/
    |-- logistic_regression.pkl
    |-- decision_tree.pkl
    |-- knn.pkl
    |-- naive_bayes.pkl
    |-- support_vector_machine.pkl
    |-- label_encoder.pkl
    |-- metadata.json
    |-- metrics_summary.csv
```

## d. Models used
Comparison of the 5 models on the test set:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9192 | 0.9934 | 0.9197 | 0.9192 | 0.9193 | 0.9023 |
| Decision Tree | 0.9070 | 0.9729 | 0.9075 | 0.9070 | 0.9071 | 0.8875 |
| KNN | 0.9136 | 0.9835 | 0.9144 | 0.9136 | 0.9138 | 0.8956 |
| Naive Bayes | 0.8970 | 0.9899 | 0.8997 | 0.8970 | 0.8972 | 0.8762 |
| Support Vector Machine | 0.9210 | 0.9933 | 0.9214 | 0.9210 | 0.9211 | 0.9045 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | TODO |
| Decision Tree | TODO |
| KNN | TODO |
| Naive Bayes | TODO |
| Support Vector Machine | TODO |
| Overall, Winner for your dataset? | TODO |

## Running locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Upload `test_data.csv` in the sidebar, pick a model, and view its metrics, confusion matrix and classification report. The app only loads the saved models; it never retrains.

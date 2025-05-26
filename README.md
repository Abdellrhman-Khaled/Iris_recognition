
# 👁️ Iris Identification using Biometrics

This MATLAB-based project implements a complete **Iris Recognition System** using various biometric techniques. It includes modules for image normalization, feature extraction (Gabor filters), classification using KNN, Decision Trees, and a Deep Learning model, as well as a performance evaluation framework.

---

## 📂 Project Structure

```
Biometrics identification/
├── main.m                         # Main entry point of the application
├── app1.mlapp                    # GUI Application
├── admin_panel.m                 # Admin panel for managing data
├── Project_documentation.pdf     # Documentation and explanation
├── Feature_extraction/
│   ├── create_gabor_kernel.m
│   ├── Gabor_features_extractor1.m
│   ├── Normalization_Daugman.m
│   └── append_feature_vector_to_excel.m
├── matching/
│   ├── compare_test.m
│   ├── detect_class_knn.m
│   ├── detect_class_trees.m
│   ├── test_deep_all.m
│   └── test_enroll.m
├── models/
│   ├── knn.m
│   ├── Trees.m
│   └── deep_model.m
└── performance/
    ├── calculate_d_prime.m
    ├── calculate_rank1.m
    ├── calculate_tmr_fmr.m
    └── compute_eer.m
```

---

## 🚀 Features

- **Iris Image Preprocessing**
  - Daugman’s Rubber Sheet Normalization
  - Gabor filter-based feature extraction

- **Classification & Matching**
  - K-Nearest Neighbors (KNN)
  - Decision Trees
  - Deep Learning (Custom network)

- **Performance Metrics**
  - Equal Error Rate (EER)
  - Rank-1 Identification Accuracy
  - TMR (True Match Rate) and FMR (False Match Rate)
  - d-prime separation

- **Admin Panel** to manage data and users
- **MATLAB App Designer Interface**

---

## 🧠 Algorithms Used

### Feature Extraction
- Gabor Filters
- Daugman’s iris normalization

### Classification
- KNN
- Decision Trees
- Shallow Deep Learning Model

### Matching Logic
- Euclidean Distance
- Class-wise Comparison
- Enrollment vs Test Matching

---

## 📊 Performance Evaluation

The system includes scripts to evaluate:
- **EER** (`compute_eer.m`)
- **Rank-1 Accuracy** (`calculate_rank1.m`)
- **d-prime value** for class separability
- TMR and FMR for security assessment

---

## 📦 Requirements

- MATLAB R2021a or later (App Designer support)
- Image Processing Toolbox
- Statistics and Machine Learning Toolbox
- Deep Learning Toolbox (for `deep_model.m`)

---

## ▶️ How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Abdellrhman-Khaled/iris-recognition.git
   cd iris-identification-biometrics
   ```

2. Open MATLAB and set the current folder to the project directory.

3. Run the main script:
   ```matlab
   main
   ```

4. Alternatively, launch the GUI:
   ```matlab
   app1.mlapp
   ```

---

## 📘 Documentation

For detailed explanation of the iris recognition pipeline, refer to the [`Project_documentation.pdf`](Biometrics%20identification/Project_documentation.pdf) file included.

---

## 📸 Screenshots

_Add screenshots here (optional GUI, performance graphs, etc.)_

---

## 🧑‍💻 Author

Email: abdellrahmankhaledd@gmail.com 
GitHub: (https://github.com/Abdellrhman-Khaled)

---

## 📄 License

Ain-shams-team-7 year 2025

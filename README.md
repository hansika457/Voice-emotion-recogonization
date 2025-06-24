  ## Emotion Recognition from Speech Using Deep Learning

This project focuses on classifying **human emotions** from **speech audio recordings** using a combination of **classical machine learning** and **deep learning** techniques. It leverages the RAVDESS dataset — a standardized audio-visual database with emotional speech and song recordings from professional actors.

The objective is to accurately identify emotions such as **Angry, Calm, Happy, Sad, Fearful, Surprised, Neutral**, and (initially) **Disgust**, based on audio features like MFCCs. After model comparison and refinement, a **CNN model** trained with **class weights** and **class balancing** was selected as the final solution.

## Dataset: RAVDESS

* **Full Name**: Ryerson Audio-Visual Database of Emotional Speech and Song
* **Format**: `.wav` files from 24 actors (12 male, 12 female) expressing emotions
* **Modalities Used**: Speech and song
* **Source**: [Official Site](https://zenodo.org/record/1188976)

##  Preprocessing & Feature Extraction

1. **Data Preparation**

   * Extracted `.wav` files from both `speech` and `song` folders.
   * Parsed file names to extract emotion labels.
   * Built a structured dataset with file paths and mapped emotion labels.

2. **Audio Feature Extraction**

   * Extracted **MFCCs** (40 coefficients), **Chroma**, and **Spectral Contrast** using `librosa`.
   * Features were aggregated to create fixed-length input vectors for ML models and 2D tensors for deep learning.

3. **Data Cleaning**

   * Normalized feature values using `StandardScaler`.
   * Label encoded the categorical emotion labels.
   * Split into training and test sets (80/20) using stratified sampling.

## Model Performance Summary

###  Classical ML Model Insights

* **Random Forest** performed the best among classical models with **71% accuracy**, especially strong on *calm* and *angry*.
* **KNN** achieved **56% accuracy**, with some reliability but struggled on underrepresented classes.
* **SVM** failed to generalize, producing only **30% accuracy** and poor recall across almost all emotions.

---

### Deep Learning Model Insights

####  CNN

* **Best performing model pre-refinement** with **76% accuracy**.
* Handled dominant classes (*angry*, *calm*, *happy*) well, but performance dipped on minority classes like *neutral* and *surprised*.

####  LSTM

* Lower accuracy (**62%**), struggled with longer temporal dependencies.
* Best recall was for *surprised*, but poor performance on *sad* and *neutral* reduced its viability.

#### CNN + LSTM Hybrid

* Balanced architecture with **72% accuracy**.
* Strong on *calm* and *surprised*, but struggled with *disgust* and *fearful*.

---

###  Final Selected Model: CNN (Refined)

**Modifications Made:**

* **Dropped "Disgust" class** due to:

  * Very limited sample size (192)
  * Consistently poor model performance across all architectures
* ⚖ **Applied Class Weights** during training to:

  * Address class imbalance
  * Improve performance on underrepresented classes like *neutral* and *surprised*

####  Final Test Results:

| Metric                    | Value    |
| ------------------------- | -------- |
| **Accuracy**              | **83%**  |
| **Macro Avg F1-Score**    | **0.83** |
| **Weighted Avg F1-Score** | **0.83** |

#### Final Test Classification Report:

```
              precision    recall  f1-score   support

       angry       0.99      0.89      0.94        75
        calm       0.83      0.93      0.88        75
     fearful       0.66      0.89      0.76        75
       happy       0.95      0.75      0.84        75
     neutral       0.94      0.76      0.84        38
         sad       0.75      0.79      0.77        75
   surprised       0.90      0.69      0.78        39
```

---

##  Final Model Architecture (CNN)

```python
Conv1D(128, kernel_size=5, activation='relu') → BatchNorm → MaxPool → Dropout(0.3)
Conv1D(256, kernel_size=5, activation='relu') → BatchNorm → MaxPool → Dropout(0.3)
Conv1D(512, kernel_size=5, activation='relu') → BatchNorm → MaxPool → Dropout(0.3)
Flatten → Dense(256) → Dropout(0.4) → Dense(128) → Dropout(0.4) → Dense(#classes, softmax)
```

* **Loss Function**: Categorical Crossentropy
* **Optimizer**: Adam (lr=0.001)
* **Callbacks**: EarlyStopping, ReduceLROnPlateau
* **Training Epochs**: \~30 (with early stopping)



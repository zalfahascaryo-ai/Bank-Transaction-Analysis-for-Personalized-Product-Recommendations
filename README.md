# Bank-Transaction-Analysis-for-Personalized-Product-Recommendations
This project focuses on analyzing financial transaction data to uncover patterns in user behavior and generate personalized product recommendations. By leveraging both unsupervised and supervised learning, the system aims to extract insights that support decision-making and potentially detect suspicious activities.

This project is divided into two main parts:
* Clustering: to identify hidden patterns and segment transaction behaviors
* Classification: to predict the costumer behavior based on history transaction

This dual approach allows both exploratory analysis and predictive modelling within the same pipeline

## **🛠️ Tech Stack**
* Python
* Pandas, Numpy (data processing)
* Scikit-learn (ML models & evaluation)
* Matplotlib, seaborn (Data visualization)
* Google Colab

## **⚙️ System Pipeline**
### **1. Exploratory Data Analysis (EDA)**
EDA is performed to understand the dataset and guide preprocessing decisions:
* Correlation Heatmap: identifies relationships between features
* Distribution Analysis: examines categorical and numerical feature distributions
* Outlier Detection: uses boxplots to identify extreme values
* Feature Relationship Analysis: uses violin plots for deeper insights

### **2. Data Preprocessing***
* Handle missing values:
    * Mode for categorical features
    * Median for numerical features
* Normalize numerical features using Min-Max Scalling
* Drop irrelevant features and duplicate rows
* Encode categorical variables using Label Encoding
* Handle outliers using log transformation
* Apply binning where necessary

### **3. CLustering Model Development***
* Select key features: CustomerAge and AccountBalance
* Determine optimal number of clusters using the Elbow Method
* Apply K-Means clustering
* Evaluate clustering using silhouette score and cluster centroid

  > PCA-based clustering was also tested and resulted in slightly higher silhouette scores.

### **4. Dataset Inversion**
To prepare for classification:
* Reverse numerical scaling back to original values
* Decode categorical features
* Merge processed datasets

### **5. Cluster Interpretation**
* Perform descriptive analysis on both normal and PCA-based clusters
* Extract insights to understand the characteristics and needs of each customer segment

### **6. Data Preparation for Classification Model**
* Remove irrelevant features
* Encode categorical variables
* Split dataset into training and testing sets (80:20 ratio)

### **7. Classification Model Development***
Multiple models are trained and compared:
* Decision Tree
* Random Forest
* Support Vector Machine (SVM)
* K-Nearest Neighbor (KNN)

### **8. Models Evaluation**
Models are evaluated using:
* Confusion Matrix
* Accuracy
* Precision
* Recall
* F1-score

### **9. Hyperparameter Tuning**
* KNN initially showed lower performance
* GridSearchCV is used to optimize hyperparameters and improve results
  
## **📊 Results**
### **Clustering**
* Clustering with K-means algorithm can distinct transcation behavior.
* Detailed insights for each cluster are provided in the Final Analysis file

### **Classification**
  * KNN (after tuning):
    <img width="444" height="393" alt="image" src="https://github.com/user-attachments/assets/5062f32e-0f7b-4240-aef4-86e2d47ee08b" />
    `Accuracy : 0.7893
    Precision: 0.7882
    Recall   : 0.7893
    F1-Score : 0.7886`

  * Decision Tree:
    <img width="444" height="393" alt="image" src="https://github.com/user-attachments/assets/02feef42-1be3-4507-a4f7-5730f628dad5" />
    `Accuracy : 0.982
    Precision: 0.982
    Recall   : 0.982
    F1-Score : 0.982`

  * Random Forest:
    <img width="444" height="393" alt="image" src="https://github.com/user-attachments/assets/d080971e-2266-4d07-9cc8-8a1f2efbdaa8" />
  `Accuracy : 0.976
  Precision: 0.976
  Recall   : 0.976
  F1-Score : 0.976`
  * SVM:
    <img width="450" height="393" alt="image" src="https://github.com/user-attachments/assets/ab3e7d42-3b66-4227-b56f-c7529c23e2fd" />
`Accuracy : 0.746
Precision: 0.745
Recall   : 0.746
F1-Score : 0.745`

## **🚀 Future Work**
* Advanced Models: Use more powerful models such as CGBoost or deep learning approach
* Anomaly Detection: Integrate anomaly detection methods to enhance fraud detection capabilities
* Deployment and feature engineering: Develop a dashboard or API for real-world applications, and incorporate time-series behavior and user profiling for deeper insights

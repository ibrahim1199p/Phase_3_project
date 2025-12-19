# 1 Business Understanding 
## 1.1 Business Overview
The telecommunications industry is one of the most competitive sectors worldwide, where customer retention plays a critical role in profitability and long-term sustainability. While acquiring new customers is costly, retaining existing ones is significantly more profitable (Propello, 2024).Globally, churn rates in telecom average around 31% annually, with mobile churn around 20%, highlighting the magnitude of the challenge.

Within Syria, the mobile telecom market is dominated by two key players: Syriatel and MTN-Syria. Syriatel, founded in 2000, currently holds about 71% of the market share and reported 20% revenue growth in 2019, equivalent to SYP 221bn (~US$242m) . Despite its dominance, Syriatel faces challenges from economic sanctions, instability, and increasing customer expectations for service quality, pricing, and personalization.
(The Syria Report, 2020),For this reason, understanding what factors drive churn and how to reduce it is a key business priority.

Churn in this context refers to the proportion of customers who stop using Syriatel’s services, either by terminating contracts or switching to competitors. Understanding and reducing churn is crucial for Syriatel’s financial health and market leadership.
## 1.2 Problem Statement
Syriatel is experiencing customer churn that directly threatens its revenues and market position. While it has historically maintained a strong market share, rising competition, service quality issues, evolving customer demands, and broader political-economic instability increase the likelihood of customer attrition. Without effective churn prediction and retention strategies, Syriatel risks losing valuable customers, resulting in revenue loss, reduced market share, and diminished competitiveness.
## 1.3 Business Objective
### 1.3.1 Main objective:
To develop a machine learning classifier that predicts whether a Syriatel customer is likely to churn, enabling data-driven strategies for proactive retention.
### 1.3.2 Specific objectives:
1. To explore customer demographics and usage behaviour influencing churn.
2. To determine how charges influence customer churn
3. To develop and evaluate machine learning models that classify whether a customer is likely to churn.
4. optimize the models for best perfomance.
5. To provide actionable insights that support Syriatel in designing targeted retention campaigns (e.g., loyalty programs, personalized offers).
## 1.4 Research Questions
1. how does customer demographics and usage behaviour influence churn?
2. how charges influence customer churn?
3. what machine learning models best predict whether a customer is likely to churn.
4. Which optimization techniques and modeling approaches most effectively improve the predictive performance of machine learning model
5. How can predictive insights be applied to practical retention strategies to minimize churn?
## 1.5 Success Criteria
- Model Performance: Achieve at least 85% accuracy and a high AUC score (>0.85) in predicting churn.
- Business Impact: Provide insights that reduce churn rates by enabling proactive retention strategies, targeting high-risk customers before they leave.

## 2. Data Understanding
### 2.1 Data overview
The dataset is from kaggle with 3333 rows aand 21 columns

### Key Attributes
1. `state`: The state of the customer.
2. `account length`: The length of the account in days .
3. `area code`: The area code of the customer's phone number.
4. `phone number`: The phone number of the customer.
5. `international plan`: Whether the customer has an international plan or not.
6. `voice mail plan`: Whether the customer has a voicemail plan or not.
7. `number vmail messages`: The number of voicemail messages the customer has.
8. `total day minutes`: Total minutes of day calls.
9. `total day calls`: Total number of day calls.
10. `total day charge`: Total charge for the day calls.
11. `total eve minutes`: Total minutes of evening calls.
12. `total eve calls`: Total number of evening calls.
13. `total eve charge`: Total charge for the evening calls.
14. `total night minutes`: Total minutes of night calls.
15. `total night calls`: Total number of night calls.
16. `total night charge`: Total charge for the night calls.
17. `total intl minutes`: Total minutes of international calls.
18. `total intl calls`: Total number of international calls.
19. `total intl charge`: Total charge for the international calls.
20. `customer service calls`: Number of times the customer called customer service.
21. `churn`: Whether the customer churned or not (True/False).


## 3. Data Preparation
### 3.1. Data Cleaning
```python
# Import libraries
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings("ignore")
```

```python
# read the data using pandas dataframe
df = pd.read_csv("Syria_tel.csv")

- the dataset had outliers which we chose to keep for purpose of analysis.
- the dataset was pretty clean with no missing values or duplicates.

## 4. EXPLORATORY DATA ANALYSIS

#plotting the class distribution using pie chart showing their proportions
plt.figure(figsize=(6,6))
df_encoded['churn'].value_counts().plot.pie(colors=['blue', 'orange'], labels=['No Churn', 'Churn'], explode=[0, 0.1],autopct='%1.1f%%' )
plt.title('Churn Distribution')
plt.ylabel("")
plt.show()
```   
### Class Distribution
![Class Distribution](./images/class_distribution.jpeg)


    
##### Customers who stayed: is 85.50% while customers who churned is 14.49%

```python
# We visualize different features against churn
features = [
    'total_day_minutes', 'total_day_calls', 'total_day_charge',
    'total_eve_minutes', 'total_eve_calls', 'total_eve_charge',
    'total_night_minutes', 'total_night_calls', 'total_night_charge',
    'total_intl_minutes', 'total_intl_calls', 'total_intl_charge',
    'customer_service_calls'
]

# Visualize boxplots for each feature against churn
plt.figure(figsize=(15, 8))
for i, col in enumerate (features[:6], 1):
    plt.subplot(2, 3, i)
    sns.boxplot(data=df, x='churn', y=col, palette=["blue", "yellow"])
    plt.title(f"{col} vs Churn")
plt.tight_layout()
plt.show()
```  
### Feature vs Churn
![Feature vs Churn](./images/Feature_vs_churn.jpeg)
 
# Modelling
#### Baseline logistic regression model vs Logistic regression model
- In comparison to our baseline model the logistic regression model with more features has an improvement in the accuacy score of 76% compared to the baseline that had 75 %.
our model is perfoming well on the 1 class but on the 0 class the model perfoms poorly
 
- Macro average (treating both classes equally): Precision = 0.64, Recall = 0.75, F1 = 0.66 → Shows imbalance in performance between the classes.

- Weighted average (accounts for class sizes): Precision = 0.86, Recall = 0.76, F1 = 0.79 → Looks better, but that’s mainly because class 0 dominates

- Despite the higher accuracy of 75% ,a precision_recall_curve of 0.29 the model is perfoming poorly we need to improve the precision.

# Decision tree

- modelling with decision tree to notice if the model perfoms better that our logistic regression.

This output indicates the top 10 features of importance in our model,with customer service calls being the top feature .

customer_service_calls (0.1679): The number of times a customer called for service is the second most important feature, playing a significant role in the prediction.

The model's decision-making process is heavily influenced by the number of customer service calls and totalday charge. Features related to international activity are also quite important. On the other hand, call counts and minutes spent during the night and evening are largely irrelevant to the model's predictions.

# modeling with random forest
- The baseline random forest model is performing better with an accuracy of 92.3% and a ROC-AUC of 0.886.
The precision o.93 and recall 0.98  for non-churners  is great resulting in F1-score of 0.96, but for churners the model struggled having a precision of 0.86 and recall of 0.57 leading to a low F1-score of 0.68
- Based on this perfomance the model we need some improvements by perfoming model tuning

# tuning hyperparameters in our model using RandimizedsearchCV

#### comparing the baseline model with the model after performing randomized search
  Model Comparison:
    Baseline RF -> Accuracy: 0.9235, ROC-AUC: 0.8862
    Tuned RF    -> Accuracy: 0.9220, ROC-AUC: 0.8963
    
- based on this comparison the tuned model has achieved an accuracy of 92% and an improved ROC-AUC of 0.896,showing better generalization.
- this is great perfomance but could also mean the model is overfitting
- to check for overfittinng we will do a cross validation to ensure our model is not overfitting

- From the cross validation results the tuned random forest model perfomed on the 5 folds shows the model perfoms well consistently in all the folds with a mean ROC-AUC of 0.9189 , this is an improvement compared to the baseline ROC-AUC of 0.886.
 
- conclusion the tuned model perfoms  better and is less likely to overfit ,compared to the baseline model

- based on this comparison the tuned model has achieved an accuracy of 92% and an improved ROC-AUC of 0.896,showing better generalization.
- this is great perfomance but could also mean the model is overfitting
- to check for overfittinng we will do a cross validation to ensure our model is not overfitting

# Further hyperparameter tuning using grid search

The grid search model improved the accuracy to 93% with ROC-AUC of 0.895.
with the crosss validation the grid is perfoming better compared to other models.
Tuning provided some improvements in our model.

from sklearn.model_selection import learning_curve

# Definig  the models
models = {
    "Baseline RF": rf_model,
    "RandomizedSearch RF": best_rf,
    "GridSearch RF": best_rf_2
}

plt.figure(figsize=(18, 5))

for i, (name, model) in enumerate(models.items(), 1):
    # Getting learning curve data
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, scoring="roc_auc", n_jobs=-1
    )
    # Computing mean scores
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    
    plt.subplot(1, 3, i)
    plt.plot(train_sizes, train_mean, "o-", label="Training AUC")
    plt.plot(train_sizes, test_mean, "o-", label="Validation AUC")
    plt.title(f"Learning Curve: {name}")
    plt.xlabel("Training examples")
    plt.ylabel("ROC-AUC")
    plt.ylim(0.7, 1.05)
    plt.legend()

plt.tight_layout()
plt.show()
   
### Random Forest Models Comparison
![Random Forest Models](./images/random_forest_models_comparison.jpeg)


From the visual all models perfoms slightly the same but the with grid search the models perfomance is doing better that the rest of the models.

## Model Evaluation
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

plt.figure(figsize=(7,5))

for name, (y_pred, y_proba) in models.items():
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc_score:.3f})")

plt.plot([0,1], [0,1], linestyle='--', color='gray')  # random guess line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves: Logistic vs Decision Tree vs Random Forest")
plt.legend()
plt.grid(True)
plt.show()
   
### Models ROC Curve
![Models ROC](./images/models_ROC.jpeg)

## Recommendations
#### 1. Adopt Random Forest as the Primary Model

- Random Forest achieved the highest ROC_AUC  of 0.896 and Average Precision of 0.0,706, making it the most reliable choice for predicting churn.

- <b>Business impact</b>: It can accurately flag customers most at risk, by an accuracy of 93%

#### 2. Use Logistic Regression When Interpretability Matters

- Logistic Regression performed weaker with an ROC_AUC of  0.815 and an  Average Precision of 0.452, but it’s easy to interpret

- <b>Recommendation</b>: Use it alongside Random Forest in scenarios where transparency and stakeholder trust are more important than raw performance.

#### 3. Treat Decision Trees as a Supporting Model

- Decision Tree had and ROC_AUC of  0.893 and and Average Precision of  0.797. These  were solid, but still below Random Forest.

- <b>Recommendation</b>: They can be useful as a simple, explainable baseline, but not as the main production model.
Monitor and retrain regularly using updated datasets to ensure the model adapts to evolving customer behavior.

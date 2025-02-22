import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_curve, roc_auc_score, auc, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, log_loss
from sklearn.model_selection import learning_curve, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from econml.dml import LinearDML




###### Assume df contains user-level data
# Normalize data
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(df.drop(['variant_A','points_earned'], axis=1))
x_scaled = pd.DataFrame(df, columns=df.columns)


# Split train and test set
X = x_scaled
y = df['variant_A']   # this is treatment assginment
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1025)


# Train treatment prediction model using 10-fold CV
rf_clf = RandomForestClassifier(criterion='gini', random_state=1025)
rf_clf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3,4,5,6],
    'max_features': [0.2, 0.4, 0.6, 0.8]
}
clf_cv = GridSearchCV(rf_clf, rf_clf_params, scoring='roc_auc', cv=10, return_train_score=True)
clf_cv.fit(X_train, y_train)

# Select the best classifier with the set of parameters that give best AUC score, assuming it is the following
best_rf_clf = RandomForestClassifier(n_estimators=300, max_depth=6, 
                                     max_features=0.6, criterion='gini', random_state=1025)
best_rf_clf.fit(X_train, y_train)
y_pred_prob = best_rf_clf.predict_proba(X_test)

# Check the performance of this classifier
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:,1])
auc_score = auc(fpr, tpr)
print('Accuracy Score: ', round(accuracy_score(y_test, y_pred),3))
print('Precision Score: ', round(precision_score(y_test, y_pred),3))
print('Recall Score: ', round(recall_score(y_test, y_pred),3))
print('F1 Score: ', round(f1_score(y_test, y_pred), 3))

# Predict probability of being in Variant A
X_train["treatment_prob"] = best_rf_clf.predict_proba(X_train)[:, 1]





# Features for predicting outcome (excluding treatment)
X2 = x_scaled
y2 = df['points_earned']   # this is outcome variable
X_train, X_test, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.3, random_state=1025)


# Train outcome model
rf_reg = RandomForestRegressor(criterion='friedman_mse', random_state=1025)
rf_reg_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3,4,5,6],
    'max_features': [0.2, 0.4, 0.6, 0.8]
}
reg_cv = GridSearchCV(rf_reg, rf_reg_params, scoring='neg_mean_squared_error', cv=10, return_train_score=True)
reg_cv.fit(X_train, y_train2)


# Select the best regressor with the set of parameters that give best AUC score, assuming it is the following
best_rf_reg = RandomForestRegressor(n_estimators=200, max_depth=6, max_features=0.6, 
                                     criterion='friedman_mse', random_state=1025)
best_rf_reg.fit(X_train, y_train2)
y_pred2 = best_rf_reg.predict(X_test)


# Check the performance of this regressor
print('Mean Absolute Error: ', round(mean_absolute_error(y_test, y_pred), 3))
print('Mean Squared Error: ', round(mean_squared_error(y_test, y_pred), 3))
print('R2: ', round(r2_score(y_test, y_pred), 3))
adj_r2 = 1 - ((1-r2_score(y_test, y_pred))*(X.shape[0]-1)/(X.shape[0]-X.shape[1]-1))
print('Adjusted R2: ', round(adj_r2, 3))

# Predict residuals of outcome
X_train["outcome_residual"] = y_train2 - best_rf_reg.predict(X_train)





# Define Double ML model
dml = LinearDML(model_y=best_rf_reg,
                model_t=best_rf_clf,
                random_state=1025)

# Fit model using confounders
dml.fit(Y=train["points_earned"], 
        T=train["Variant_A"], 
        X=train[["past_purchases", "app_sessions", "user_tenure"]],
        W=None)  # No extra controls

# Get treatment effect estimates
treatment_effects = dml.effect(train[["past_purchases", "app_sessions", "user_tenure"]])

# Analyze effects by segment
train["treatment_effect"] = treatment_effects
train.groupby("user_segment")["treatment_effect"].mean()


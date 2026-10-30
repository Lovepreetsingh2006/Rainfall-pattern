import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

S0 = pd.read_csv(r"D:\py\rainfalldata.csv")


#STEP 1 1-DATA CLEANING
print(S0.head(10))
print(S0.info())
print(S0.describe())

#DATA CLEANING
print(S0.isnull().sum())
S0.fillna(S0.mean(numeric_only=True), inplace=True)
print(S0.isnull().sum())
print(S0.info())

#STEP 2- Feature Adding and Engg.
monthly_cols = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
S0['TOTAL_CALCULATED'] = S0[monthly_cols].sum(axis=1)
S0['DIFFERENCE'] = S0['ANNUAL'] - S0['TOTAL_CALCULATED']

print(S0['DIFFERENCE'].describe())

S0['ANNUAL'] = S0[monthly_cols].sum(axis=1)
S0['Jan-Feb'] = S0['JAN'] + S0['FEB']
S0['Mar-May'] = S0['MAR'] + S0['APR'] + S0['MAY']
S0['Jun-Sep'] = S0['JUN'] + S0['JUL'] + S0['AUG'] + S0['SEP']
S0['Oct-Dec'] = S0['OCT'] + S0['NOV'] + S0['DEC']

# Rainfall intensity category
def rainfall_label(x):
    if x < 500:
        return "Low"
    elif x < 1500:
        return "Medium"
    else:
        return "High"

S0['RAIN_CATEGORY'] = S0['ANNUAL'].apply(rainfall_label)


#STEP 3- VISULIZATION
top7 = S0.groupby('SUBDIVISION')['ANNUAL'].mean().sort_values(ascending=False).head(7)
top7_df = top7.reset_index()
plt.figure(figsize=(10,5))
sns.barplot(data=top7_df,x='ANNUAL',y='SUBDIVISION',color='Red')
plt.title("Top 7 High Rainfall States (Flood Risk Zones)")
plt.xlabel("Average Annual Rainfall")
plt.ylabel("State")
plt.show()
# Conclusion: Coastal Karnataka and Arunachal Pradesh receive the highest rainfall,
# indicating strong flood risk zones. Most high rainfall regions are coastal or northeastern areas.

yearly_avg = S0.groupby('YEAR')['ANNUAL'].mean()
plt.figure(figsize=(10,5))
sns.lineplot(x=yearly_avg.index, y=yearly_avg.values)
plt.title("Average Annual Rainfall Trend")
plt.show()
# Conclusion: Rainfall fluctuates over the years with no consistent upward or downward trend,
# showing variability but overall stable long-term average.


plt.figure(figsize=(8,5))
sns.histplot(S0['ANNUAL'], bins=30, kde=False)
plt.title("Distribution of Annual Rainfall")
plt.xlabel("Rainfall")
plt.ylabel("Frequency")
plt.show()






season_cols = ['Jan-Feb','Mar-May','Jun-Sep','Oct-Dec']
season_mean = S0[season_cols].mean()
plt.figure(figsize=(8,4))
sns.barplot(x=season_mean.index,y=season_mean.values,palette='viridis')
plt.title("Average Seasonal Rainfall")
plt.ylabel("Rainfall")
plt.xlabel("Season")
plt.show()
# Conclusion: Most rainfall values are concentrated in the mid-range,
# with fewer extreme high rainfall events, indicating a slightly right-skewed distribution.



monthly_avg = S0[monthly_cols].mean()
plt.figure(figsize=(8,4))
sns.lineplot(x=monthly_avg.index, y=monthly_avg.values, marker='o')
plt.title("Average Monthly Rainfall Pattern")
plt.xlabel("Month")
plt.ylabel("Rainfall")
plt.show()



sns.countplot(x='RAIN_CATEGORY', data=S0)
plt.title("Rainfall Category Distribution")
plt.show()
# Conclusion: Most observations fall under the medium rainfall category,
# with fewer instances of high and low rainfall.


matrix = S0[['ANNUAL', 'Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec']].corr()
plt.figure(figsize=(6, 4))
sns.heatmap(matrix,annot=True,cmap='coolwarm',linewidths=0.5)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.title('Heatmap of Annual and Seasonal Rainfall')
plt.show()
# Conclusion: Annual rainfall is highly correlated with Jun-Sep rainfall,
# indicating monsoon season is the key contributor to total yearly rainfall.




plt.figure(figsize=(6,4))
sns.scatterplot(x=S0['Jun-Sep'], y=S0['ANNUAL'])
plt.title("Jun-Sep vs Annual Rainfall")
plt.xlabel("Monsoon Rainfall")
plt.ylabel("Annual Rainfall")
plt.show()
# Conclusion: There is a strong positive relationship between monsoon rainfall
# and annual rainfall, showing that higher monsoon rainfall leads to higher yearly totals.

# STEP 4 - MACHINE LEARNING


# STEP 4.1: Copy cleaned data into S1 (IMPORTANT)

S1 = S0.copy()
S1 = S1.sort_values(['SUBDIVISION','YEAR'])
S1['NEXT_YEAR_RAIN'] = S1.groupby('SUBDIVISION')['ANNUAL'].shift(-1)
S1.dropna(inplace=True)
    

new_cols=['ANNUAL']
outliers=[]
for col in new_cols:
    Q1=S1[col].quantile(0.25)#it marks the point below which 25% of the data falls.
    Q3=S1[col].quantile(0.75)#it marks the point below which 75% of the data falls.
    IQR=Q3-Q1
    lower_bound=Q1-1.5*IQR
    upper_bound=Q3+1.5*IQR
    outliers.extend(S1[(S1[col]<lower_bound)|(S1[col]>upper_bound)].index)
S1=S1.drop(set(outliers))

plt.figure(figsize=(6,4))
sns.scatterplot(x=S1['ANNUAL'], y=S1['NEXT_YEAR_RAIN'])
plt.title("ANNUAL vs NEXT_YEAR_RAIN")
plt.xlabel("ANNUAL Rainfall")
plt.ylabel("NEXT_YEAR Rainfall")
plt.show()


X = S1[['ANNUAL']]
Y = S1['NEXT_YEAR_RAIN']
scaler=MinMaxScaler()
X=scaler.fit_transform(X)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=34)


model = LinearRegression()
model.fit(x_train, y_train)

new_data = pd.DataFrame({'ANNUAL':[3373.2]})

new_scaled = scaler.transform(new_data)
result = model.predict(new_scaled)
print("Predicted Next Year Rainfall:", result[0])

y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.2f}")

from sklearn.metrics import r2_score
print(f"R2 Score:, {r2_score(y_test, y_pred):.2f}")

from sklearn.metrics import mean_absolute_error
print(f"Mean absolute error:, {mean_absolute_error(y_test, y_pred):.2f}")


plt.scatter(X, Y,color='blue')
plt.plot(X, model.predict(X), color='pink', linewidth=5)
plt.xlabel("ANNUAL")
plt.ylabel("NEXT_YEAR")
plt.title("Linear regression fit")
plt.show()


# ======================================================================
# STEP 5 - MULTIPLE LINEAR REGRESSION (Seasons + Region)
# ======================================================================

# Features: 4 seasonal totals + SUBDIVISION (converted to 0/1 columns)
X2 = S1[['Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec', 'SUBDIVISION']]
X2 = pd.get_dummies(X2, columns=['SUBDIVISION'], drop_first=True, dtype=int)
Y2 = S1['NEXT_YEAR_RAIN']

# Same split as simple linear regression
x2_train, x2_test, y2_train, y2_test = train_test_split(X2, Y2, test_size=0.2, random_state=34)

model2 = LinearRegression()
model2.fit(x2_train, y2_train)

y2_pred = model2.predict(x2_test)

mse2 = mean_squared_error(y2_test, y2_pred)
print(f"Mean Squared Error (MSE): {mse2:.2f}")
print(f"Root Mean Squared Error (RMSE): {np.sqrt(mse2):.2f}")
print(f"Mean absolute error: {mean_absolute_error(y2_test, y2_pred):.2f}")
print(f"R2 Score: {r2_score(y2_test, y2_pred):.2f}")

plt.scatter(y2_test, y2_pred, color='blue')
plt.plot(y2_test, y2_test, color='pink', linewidth=5)
plt.xlabel("ACTUAL NEXT_YEAR")
plt.ylabel("PREDICTED NEXT_YEAR")
plt.title("Multiple linear regression fit")
plt.show()
# Conclusion: Adding seasonal rainfall and region improves the prediction compared with
# simple linear regression, because rainfall level of a region is stable from year to year.


# ======================================================================
# STEP 6 - KNN REGRESSOR (Seasons + Region)
# ======================================================================
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score

# KNN works on distance, so scale all columns to 0-1 (fit on training data only)
scaler3 = MinMaxScaler()
x3_train = scaler3.fit_transform(x2_train)
x3_test = scaler3.transform(x2_test)

# Choose the best K using training data only (5-fold cross validation)
k_values = [15, 20, 25, 30, 40, 50]
best_k = k_values[0]
best_mse = float('inf')
for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k)
    scores = cross_val_score(knn, x3_train, y2_train, cv=5, scoring='neg_mean_squared_error')
    cv_mse = -scores.mean()
    print(f"K = {k}, CV MSE = {cv_mse:.2f}")
    if cv_mse < best_mse:
        best_mse = cv_mse
        best_k = k
print("Best K:", best_k)

model3 = KNeighborsRegressor(n_neighbors=best_k)
model3.fit(x3_train, y2_train)

y3_pred = model3.predict(x3_test)

mse3 = mean_squared_error(y2_test, y3_pred)
print(f"Mean Squared Error (MSE): {mse3:.2f}")
print(f"Root Mean Squared Error (RMSE): {np.sqrt(mse3):.2f}")
print(f"Mean absolute error: {mean_absolute_error(y2_test, y3_pred):.2f}")
print(f"R2 Score: {r2_score(y2_test, y3_pred):.2f}")

plt.scatter(y2_test, y3_pred, color='blue')
plt.plot(y2_test, y2_test, color='pink', linewidth=5)
plt.xlabel("ACTUAL NEXT_YEAR")
plt.ylabel("PREDICTED NEXT_YEAR")
plt.title("KNN regression fit")
plt.show()


# ======================================================================
# STEP 7 - MODEL COMPARISON (all models use the same train/test split)
# ======================================================================
comparison = pd.DataFrame({
    'Model': ['Simple LR', 'Multiple LR', 'KNN'],
    'MAE': [mean_absolute_error(y_test, y_pred),
            mean_absolute_error(y2_test, y2_pred),
            mean_absolute_error(y2_test, y3_pred)],
    'MSE': [mean_squared_error(y_test, y_pred),
            mean_squared_error(y2_test, y2_pred),
            mean_squared_error(y2_test, y3_pred)],
    'R2': [r2_score(y_test, y_pred),
           r2_score(y2_test, y2_pred),
           r2_score(y2_test, y3_pred)]
})
comparison['RMSE'] = np.sqrt(comparison['MSE'])
print(comparison.round(2).to_string(index=False))

plt.figure(figsize=(6,4))
sns.barplot(data=comparison, x='Model', y='RMSE', color='steelblue')
plt.title("RMSE of the Three Models (lower is better)")
plt.xlabel("Model")
plt.ylabel("RMSE")
plt.show()
# Conclusion: Multiple Linear Regression and KNN perform much better than Simple Linear
# Regression because they use seasonal rainfall and region information. MLR and KNN give
# almost the same accuracy, so MLR is the best choice as it is simpler and easier to explain.


# ======================================================================
# STEP 8 - PREDICTION EXAMPLE (same input for all three models)
# ======================================================================
# Previous year's seasonal rainfall (mm) - example values, replace with real ones
new_row = pd.DataFrame({
    'Jan-Feb': [60.0],
    'Mar-May': [90.0],
    'Jun-Sep': [450.0],
    'Oct-Dec': [35.0],
    'SUBDIVISION': ['PUNJAB']
})

# Simple LR needs only the annual total
annual_total = new_row[season_cols].sum(axis=1)[0]
pred1 = model.predict(scaler.transform(pd.DataFrame({'ANNUAL': [annual_total]})))

# MLR and KNN need the same 39 columns as training
new_X = pd.get_dummies(new_row, columns=['SUBDIVISION'], dtype=int)
new_X = new_X.reindex(columns=x2_train.columns, fill_value=0)

pred2 = model2.predict(new_X)
pred3 = model3.predict(scaler3.transform(new_X))

print("\nPredicted next year rainfall for PUNJAB:")
print("Simple Linear Regression  :", round(pred1[0], 2))
print("Multiple Linear Regression:", round(pred2[0], 2))
print("KNN Regressor             :", round(pred3[0], 2))
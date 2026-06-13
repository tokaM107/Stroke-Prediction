import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# 1. Loading the dataset
strokedf = pd.read_csv("data/raw.csv")

# 2. Dropping the id column as it is not useful for our analysis
strokedf.drop(columns=['id'], inplace=True, errors='ignore') 

# 3. Checking for missing values
print("Missing values before fix:\n", strokedf.isnull().sum())

# [تم الإصلاح] Filling the missing values in the bmi column correctly
strokedf['bmi']=strokedf['bmi'].fillna(strokedf['bmi'].median(), inplace=True)

# 4. Checking and handling duplicates
print("Total duplicates:", strokedf.duplicated().sum())
strokedf.drop_duplicates(inplace=True)

# 5. Shuffling the dataset
strokedf = strokedf.sample(frac=1, random_state=42).reset_index(drop=True)

# 6. Handling outliers using boxplot (Capping)
def handle_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1  
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
    df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    print(f"{column}: Lower Bound = {lower_bound:.4f}, Upper Bound = {upper_bound:.4f}")

handle_outliers_iqr(strokedf, 'avg_glucose_level')
handle_outliers_iqr(strokedf, 'bmi')

# 7. Fixing skewness using log transformation
strokedf['avg_glucose_level'] = np.log1p(strokedf['avg_glucose_level'])

# 8. Filtering age (Valid and Adults only)
strokedf = strokedf[strokedf['age'] >= 18]

# 9. Encoding the categorical variables using one-hot encoding
strokedf = pd.get_dummies(strokedf, columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'], drop_first=True, dtype=int)

# 10. Creating the new feature: health_factor (Done before filtering to make sure it includes columns)
# Note: since glucose is log-transformed, 140 becomes log1p(140) ~= 4.9487
strokedf['health_factor'] = (
    strokedf['hypertension'] + 
    strokedf['heart_disease'] + 
    (strokedf['avg_glucose_level'] > np.log1p(140)).astype(int) 
)

# 11. Keeping only the important columns based on insights
columns_to_keep = [
    'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
    'gender_Male', 'ever_married_Yes', 'work_type_Self-employed',
    'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes',
    'health_factor', 
    'stroke'  
]
strokedf_filtered = strokedf[columns_to_keep]

# 12. Save the cleaned dataset to data/cleaned.csv
strokedf_filtered.to_csv("data/cleaned.csv", index=False)
print("Cleaned dataset saved successfully to 'data/cleaned.csv'")

# scaler = ColumnTransformer(
#     transformers=[('scaler', StandardScaler(), ['age', 'bmi', 'avg_glucose_level'])],
#     remainder='passthrough'
# )


# X_filtered = strokedf_filtered.drop(columns=['stroke'])
# y_filtered = strokedf_filtered['stroke']

# scaled_array = scaler.fit_transform(X_filtered)

# feature_names = scaler.get_feature_names_out()
# feature_names_clean = [name.split('__')[-1] for name in feature_names]

# strokedf_preprocessed = pd.DataFrame(scaled_array, columns=feature_names_clean)
# strokedf_preprocessed['stroke'] = y_filtered.values 

# # 13. Save the preprocessed dataset to data/preprocessed.csv
# strokedf_preprocessed.to_csv("data/preprocessed.csv", index=False)
# print("🚀 Preprocessed dataset saved successfully to 'data/preprocessed.csv'")

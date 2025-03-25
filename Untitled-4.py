# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# %% [markdown]
# The aim is to build a predictive model and predict the sales of each product at a particular outlet.
# 
# Using this model, BigMart will try to understand the properties of products and outlets which play a key role in increasing sales.

# %% [markdown]
# Submission file format
# 
# <!-- Variable	Description
# Item_Identifier	Unique product ID
# Outlet_Identifier	Unique store ID
# Item_Outlet_Sales	Sales of the product in the particular store. This is the outcome variable to be predicted. -->

# %%
# Submission file format

# Variable	Description
# Item_Identifier	Unique product ID
# Outlet_Identifier	Unique store ID
# Item_Outlet_Sales	Sales of the product in the particular store. This is the outcome variable to be predicted.

# %%
data=pd.read_csv('/Users/manishmaurya/Downloads/train_v9rqX0R.csv')
test_data=pd.read_csv('/Users/manishmaurya/Downloads/test_AbJTz2l.csv')

# %%
data.info()

# %%
replacements = {
    'LF': 'Low Fat',
    'low fat': 'Low Fat',
    'reg': 'Regular'
}

# Replace the values in the 'Item_Fat_Content' column
data['Item_Fat_Content'] = data['Item_Fat_Content'].replace(replacements)
test_data['Item_Fat_Content'] = test_data['Item_Fat_Content'].replace(replacements)

# %%
# data['outlet_rel_age']=data['Outlet_Establishment_Year'].max()-data['Outlet_Establishment_Year']+1
# test_data['outlet_rel_age']=test_data['Outlet_Establishment_Year'].max()-test_data['Outlet_Establishment_Year']+1


# # Normalize Item_Visibility
# scaler = MinMaxScaler()
# test_data['Normalized_Visibility'] = scaler.fit_transform(test_data[['Item_Visibility']])

# # Encode Outlet_Size
# encoder = LabelEncoder()
# test_data['Encoded_Outlet_Size'] = encoder.fit_transform(test_data['Outlet_Size'])
# test_data['visibility_outlet'] = test_data['Normalized_Visibility'] + test_data['Encoded_Outlet_Size']

# %%
# from pandas.plotting import parallel_coordinates
# # Select columns for parallel coordinates plot
# # selected_columns = ['Item_Outlet_Sales', 'Item_Weight', 'Item_Visibility']
# selected_columns = ['Item_Weight', 'Item_Fat_Content', 'Item_Visibility', 'Item_Type', 'Item_MRP', 'outlet_rel_age', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type','visibility_outlet','Item_Outlet_Sales']


# # Encode categorical columns
# encoder = LabelEncoder()
# for col in ['Item_Type', 'Outlet_Size','Item_Fat_Content','Outlet_Location_Type', 'Outlet_Type']:
#     data[col] = encoder.fit_transform(data[col])

# # Normalize the data
# scaler = MinMaxScaler()
# data[selected_columns] = scaler.fit_transform(data[selected_columns])

# # Create a dummy class column
# data['Dummy_Class'] = 'All'

# # Plot parallel coordinates
# plt.figure(figsize=(16, 4))
# parallel_coordinates(data[selected_columns + ['Outlet_Identifier']], 'Outlet_Identifier', colormap='viridis')
# plt.title('Parallel Coordinates Plot')
# plt.xlabel('Features')
# plt.ylabel('Values')
# plt.show()

# %%
# # Based on EDA, choose relevant features (adjust based on your findings)
# selected_features = ['Item_Weight', 'Item_Fat_Content', 'Item_Visibility', 'Item_Type', 'Item_MRP', 
#                      'outlet_rel_age', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type','visibility_outlet']

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt



# 2. Basic Feature Engineering (Avoiding Leakage)
data['Outlet_Age'] = 2023 - data['Outlet_Establishment_Year']
test_data['Outlet_Age'] = 2023 - test_data['Outlet_Establishment_Year']

# 3. Handle Missing Values (Imputation on Training Data Only)
numerical_cols = data.select_dtypes(include=['number']).columns
categorical_cols = data.select_dtypes(include=['object']).columns

# Impute numerical features on training data, then apply to test data
for col in numerical_cols:
    mean_val = data[col].mean()
    data[col].fillna(mean_val, inplace=True)
    if col in test_data.columns: # Added this check
        test_data[col].fillna(mean_val, inplace=True)

# Impute categorical features on training data, then apply to test data
for col in categorical_cols:
    mode_val = data[col].mode()[0]
    data[col].fillna(mode_val, inplace=True)
    if col in test_data.columns: # Added this check
        test_data[col].fillna(mode_val, inplace=True)

# %%


# 4. Feature Selection (Keeping Only Relevant Features)
selected_features = ['Item_Weight', 'Item_Visibility', 'Item_Type', 'Item_MRP', 'Outlet_Age', 'Outlet_Type']
data_selected = data[selected_features + ['Item_Outlet_Sales']]

X = data_selected.drop('Item_Outlet_Sales', axis=1)
y = data_selected['Item_Outlet_Sales']

# 5. Data Splitting (Ensuring No Leakage)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Preprocessing (Scaling and Encoding - On Training Data Only, Then Applied to Validation/Test)
numerical_features = X_train.select_dtypes(include=['number']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

numerical_transformer = RobustScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Fit and transform the training data
X_train_processed = preprocessor.fit_transform(X_train)

# Transform the validation and test data using the fitted preprocessor
X_val_processed = preprocessor.transform(X_val)
test_processed = preprocessor.transform(test_data[selected_features])

# %%


# 7. Build ANN Model (with Regularization)
model = keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=(X_train_processed.shape[1],), kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.4),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.4),
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# 8. Train ANN and Store History
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

history = model.fit(X_train_processed, y_train, epochs=200, batch_size=32, validation_data=(X_val_processed, y_val), callbacks=[early_stopping], verbose=0)

# 9. Evaluate ANN
train_predictions = model.predict(X_train_processed).flatten()
train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
print(f'ANN Training RMSE: {train_rmse}')

# 10. Validation Plot
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()



# %%
# 11. Make Predictions on Test Data
predictions = model.predict(test_processed).flatten()

# 12. Create Submission File
submission = pd.DataFrame({'Item_Identifier': test_data['Item_Identifier'],
                           'Outlet_Identifier': test_data['Outlet_Identifier'],
                           'Item_Outlet_Sales': predictions})

submission.to_csv('submission.csv', index=False)
print('Predictions saved to submission.csv')

# 13. Data Integrity Checks
print("Target Variable Statistics:")
print(y.describe())

import seaborn as sns
plt.figure(figsize=(8, 6))
sns.histplot(y, kde=True)
plt.title('Distribution of Item_Outlet_Sales')
plt.show()

print(f"Number of duplicate rows: {data.duplicated().sum()}")

# %%




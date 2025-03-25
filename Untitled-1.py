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
test_data['Outlet_Identifier'].unique()

# %%
data['Outlet_Identifier'].unique()

# %%
test_data.groupby('Outlet_Identifier')['Item_Identifier'].unique()
data.groupby('Outlet_Identifier')['Item_Identifier'].unique()

# %%
# Get unique Item_Identifier for each Outlet_Identifier in both DataFrames
test_unique_items = test_data.groupby('Outlet_Identifier')['Item_Identifier'].unique()
data_unique_items = data.groupby('Outlet_Identifier')['Item_Identifier'].unique()

# Find common Item_Identifier for each Outlet_Identifier
common_items = {}
for outlet in test_unique_items.index:
    if outlet in data_unique_items.index:
        common_items[outlet] = set(test_unique_items[outlet]).intersection(set(data_unique_items[outlet]))

# Print the common Item_Identifier for each Outlet_Identifier
for outlet, items in common_items.items():
    print(f"Outlet {outlet} has common Item_Identifier: {items}")

# %%
data.groupby(['Outlet_Size','Outlet_Type','Outlet_Identifier'])['Item_Outlet_Sales'].sum()

# %%
data['outlet_rel_age']=data['Outlet_Establishment_Year'].max()-data['Outlet_Establishment_Year']+1
test_data['outlet_rel_age']=test_data['Outlet_Establishment_Year'].max()-test_data['Outlet_Establishment_Year']+1


# Normalize Item_Visibility
scaler = MinMaxScaler()
test_data['Normalized_Visibility'] = scaler.fit_transform(test_data[['Item_Visibility']])

# Encode Outlet_Size
encoder = LabelEncoder()
test_data['Encoded_Outlet_Size'] = encoder.fit_transform(test_data['Outlet_Size'])
test_data['visibility_outlet'] = test_data['Normalized_Visibility'] + test_data['Encoded_Outlet_Size']

# %%
data.groupby('Outlet_Identifier')['Item_MRP'].min()

# %%
data.head()

# %%
# Fill missing Item_Visibility with the mean of the corresponding Outlet_Identifier group
data['Item_Visibility'] = data.groupby('Outlet_Identifier')['Item_Visibility'].transform(lambda x: x.fillna(x.mean()))

# Fill missing Outlet_Size with the mode of the column
data['Outlet_Size'].fillna(data['Outlet_Size'].mode()[0], inplace=True)

# Normalize Item_Visibility
scaler = MinMaxScaler()
data['Normalized_Visibility'] = scaler.fit_transform(data[['Item_Visibility']])

# Encode Outlet_Size
encoder = LabelEncoder()
data['Encoded_Outlet_Size'] = encoder.fit_transform(data['Outlet_Size'])

# Combine the features
data['visibility_outlet'] = data['Normalized_Visibility'] + data['Encoded_Outlet_Size']
data['unit_wt_MRP']=data['Item_MRP']/data['Item_Weight']

# %%
data.head()

# %%


# %%


# %%
def range_func(x):
    return max(x) - min(x)

data.pivot_table(index='Item_Identifier', columns='Outlet_Identifier', values='Item_MRP', aggfunc=range_func)

# %%
data.info()

# %%
data['Outlet_Establishment_Year'].min()

# %%
data['Item_Type'].value_counts()

# %%
sns.pairplot(data,x_vars=['Item_Weight', 'Item_Visibility', 'Item_MRP', 'visibility_outlet'],y_vars=['Item_Outlet_Sales','Item_Weight'],hue='outlet_rel_age')

# %%


# %%
id_attributes=['Item_Identifier','Outlet_Identifier']

# %%
categorical_features=['Item_Fat_Content','Item_Type','Outlet_Establishment_Year','Outlet_Size','Outlet_Location_Type','Outlet_Type']

# %%
# Get the unique years in sorted order
years_order = sorted(data['Outlet_Establishment_Year'].unique())

# Create the bar plot with sorted x-axis
sns.barplot(data=data, x='outlet_rel_age', y='Item_Outlet_Sales')

# %%
sns.barplot(data=data, y='Item_Type', x='Item_Outlet_Sales')

# %%
data.head()

# %%
# Based on EDA, choose relevant features (adjust based on your findings)
selected_features = ['Item_Weight', 'Item_Fat_Content', 'Item_Visibility', 'Item_Type', 'Item_MRP', 
                     'outlet_rel_age', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type','visibility_outlet']

# %%
train_data=data.copy()

# %%
# 2.2. Numerical Feature Analysis
numerical_features = data.select_dtypes(include=['number']).columns
# for feature in numerical_features:
#     plt.figure()
#     sns.scatterplot(x=feature, y='Item_Outlet_Sales', data=data)
#     plt.title(f'{feature} vs. Item_Outlet_Sales')
#     plt.show()

# %%
# 2.3. Categorical Feature Analysis
categorical_features = data.select_dtypes(include=['object']).columns.drop(['Item_Identifier', 'Outlet_Identifier']) # Exclude IDs

for feature in categorical_features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=feature, y='Item_Outlet_Sales', data=data)
    plt.title(f'{feature} vs. Item_Outlet_Sales')
    plt.xticks(rotation=45)
    plt.show()

# %%
data.info()

# %%
# 3. Data Preprocessing
# data['Item_Weight'].fillna(data['Item_Weight'].mean(), inplace=True)

# Fill missing Item_Weight with the mean of the corresponding Item_Identifier group
data['Item_Weight'] = data.groupby('Item_Identifier')['Item_Weight'].transform(lambda x: x.fillna(x.mean()))
data['Outlet_Size'].fillna(data['Outlet_Size'].mode()[0], inplace=True)



# %%
# Based on EDA, choose relevant features (adjust based on your findings)
selected_features = ['Item_Weight', 'Item_Fat_Content', 'Item_Visibility', 'Item_Type', 'Item_MRP', 
                     'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type','outlet_rel_age','visibility_outlet']

# %%


# %%


# %%
data_selected.info()

# %%

data_selected = data[selected_features + ['Item_Outlet_Sales']]
data_selected.dropna(inplace=True)


X = data_selected.drop('Item_Outlet_Sales', axis=1)
y = data_selected['Item_Outlet_Sales']

categorical_selected = X.select_dtypes(include=['object']).columns
numerical_selected = X.select_dtypes(include=['number']).columns

numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_selected),
        ('cat', categorical_transformer, categorical_selected)
    ])


# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Model Selection and Evaluation (Linear vs. Non-Linear)
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

results = {}
for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = (-scores)**0.5
    results[name] = rmse_scores.mean()

print("Model Evaluation:")
for name, rmse in results.items():
    print(f"{name}: RMSE = {rmse:.4f}")


# 6. Choose the Best Model (Based on Results)
best_model_name = min(results, key=results.get)
print(f"\nBest Fitted Model: {best_model_name} with RMSE = {results[best_model_name]:.4f}")



# 7. Hyperparameter Tuning (Example with Best Model - Gradient Boosting)
if best_model_name == "Gradient Boosting":  # Adjust if a different model is chosen
    pipeline_best = Pipeline(steps=[('preprocessor', preprocessor), ('model', GradientBoostingRegressor(random_state=42))])
    param_grid_best = {
        'model__n_estimators': [100, 200, 300],
        'model__learning_rate': [0.01, 0.05, 0.1],
        'model__max_depth': [3, 4, 5]
    }
    grid_search_best = GridSearchCV(pipeline_best, param_grid_best, cv=5, scoring='neg_mean_squared_error')
    grid_search_best.fit(X_train, y_train)
    best_model = grid_search_best.best_estimator_
else:
    best_model = Pipeline(steps=[('preprocessor', preprocessor), ('model', models[best_model_name])])
    best_model.fit(X_train, y_train)

# %%
# !pip install xgboost

# %%
data.info()

# %%


# 4. Feature Selection (If needed)
selected_features = ['Item_Weight', 'Item_Fat_Content', 'Item_Visibility', 'Item_Type', 'Item_MRP', 'outlet_rel_age', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']
data_selected = data[selected_features + ['Item_Outlet_Sales']]

X = data_selected.drop('Item_Outlet_Sales', axis=1)
y = data_selected['Item_Outlet_Sales']

categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=['number']).columns

numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False) #Added sparse_output=False

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 5. Model Training and Hyperparameter Tuning (HistGradientBoostingRegressor)
pipeline_hgb = Pipeline(steps=[('preprocessor', preprocessor), ('model', HistGradientBoostingRegressor(random_state=42))])
param_grid_hgb = {
    'model__max_iter': [100, 200, 300],
    'model__learning_rate': [0.01, 0.05, 0.1],
    'model__max_depth': [3, 4, 5]
}

grid_search_hgb = GridSearchCV(pipeline_hgb, param_grid_hgb, cv=5, scoring='neg_mean_squared_error')
grid_search_hgb.fit(X, y)

best_hgb = grid_search_hgb.best_estimator_

# 6. Make Predictions (If needed)
# ...

# %%
# 6. Make Predictions on Test Data

# 8. Make Predictions on Test Data (if you have one)
# test_data1 = pd.read_csv('test_data1.csv')  # Uncomment if you have test data
test_data1 = test_data[selected_features +['Item_Identifier']+['Outlet_Identifier']] # Select chosen features

# test_data1['Item_Weight'].fillna(test_data1['Item_Weight'].mean(), inplace=True)
test_data1['Item_Weight'] = test_data1.groupby('Item_Identifier')['Item_Weight'].transform(lambda x: x.fillna(x.mean()))
# Fill any remaining NaN values with the overall mean
test_data1['Item_Weight'].fillna(test_data1['Item_Weight'].mean(), inplace=True)

test_data1['Outlet_Size'].fillna(test_data1['Outlet_Size'].mode()[0], inplace=True)
predictions = best_hgb.predict(test_data1)

# 7. Create Submission File
submission = pd.DataFrame({'Item_Identifier': test_data['Item_Identifier'],
                           'Outlet_Identifier': test_data['Outlet_Identifier'],
                           'Item_Outlet_Sales': predictions})

submission.to_csv('submission.csv', index=False)
print('Predictions saved to submission.csv')



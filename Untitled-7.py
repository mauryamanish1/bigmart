{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "# !pip install --upgrade xgboost"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    " # !pip install matplotlib"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "# import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import pandas as pd\n",
    "from sklearn.preprocessing import MinMaxScaler, LabelEncoder\n",
    "from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV\n",
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV\n",
    "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.ensemble import GradientBoostingRegressor\n",
    "from sklearn.metrics import mean_squared_error\n",
    "\n",
    "\n",
    "from sklearn.model_selection import train_test_split, GridSearchCV\n",
    "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.ensemble import HistGradientBoostingRegressor\n",
    "from sklearn.metrics import mean_squared_error\n",
    "\n",
    "\n",
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.linear_model import LinearRegression, Ridge, Lasso\n",
    "from sklearn.tree import DecisionTreeRegressor\n",
    "from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor\n",
    "from sklearn.metrics import mean_squared_error"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The aim is to build a predictive model and predict the sales of each product at a particular outlet.\n",
    "\n",
    "Using this model, BigMart will try to understand the properties of products and outlets which play a key role in increasing sales."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Submission file format\n",
    "\n",
    "<!-- Variable\tDescription\n",
    "Item_Identifier\tUnique product ID\n",
    "Outlet_Identifier\tUnique store ID\n",
    "Item_Outlet_Sales\tSales of the product in the particular store. This is the outcome variable to be predicted. -->"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Submission file format\n",
    "\n",
    "# Variable\tDescription\n",
    "# Item_Identifier\tUnique product ID\n",
    "# Outlet_Identifier\tUnique store ID\n",
    "# Item_Outlet_Sales\tSales of the product in the particular store. This is the outcome variable to be predicted."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "data=pd.read_csv('/Users/manishmaurya/Downloads/train_v9rqX0R.csv')\n",
    "test_data=pd.read_csv('/Users/manishmaurya/Downloads/test_AbJTz2l.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 8523 entries, 0 to 8522\n",
      "Data columns (total 12 columns):\n",
      " #   Column                     Non-Null Count  Dtype  \n",
      "---  ------                     --------------  -----  \n",
      " 0   Item_Identifier            8523 non-null   object \n",
      " 1   Item_Weight                7060 non-null   float64\n",
      " 2   Item_Fat_Content           8523 non-null   object \n",
      " 3   Item_Visibility            8523 non-null   float64\n",
      " 4   Item_Type                  8523 non-null   object \n",
      " 5   Item_MRP                   8523 non-null   float64\n",
      " 6   Outlet_Identifier          8523 non-null   object \n",
      " 7   Outlet_Establishment_Year  8523 non-null   int64  \n",
      " 8   Outlet_Size                6113 non-null   object \n",
      " 9   Outlet_Location_Type       8523 non-null   object \n",
      " 10  Outlet_Type                8523 non-null   object \n",
      " 11  Item_Outlet_Sales          8523 non-null   float64\n",
      "dtypes: float64(4), int64(1), object(7)\n",
      "memory usage: 799.2+ KB\n"
     ]
    }
   ],
   "source": [
    "data.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "replacements = {\n",
    "    'LF': 'Low Fat',\n",
    "    'low fat': 'Low Fat',\n",
    "    'reg': 'Regular'\n",
    "}\n",
    "\n",
    "# Replace the values in the 'Item_Fat_Content' column\n",
    "data['Item_Fat_Content'] = data['Item_Fat_Content'].replace(replacements)\n",
    "test_data['Item_Fat_Content'] = test_data['Item_Fat_Content'].replace(replacements)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "data['outlet_rel_age']=data['Outlet_Establishment_Year'].max()-data['Outlet_Establishment_Year']+1\n",
    "# test_data['outlet_rel_age']=test_data['Outlet_Establishment_Year'].max()-test_data['Outlet_Establishment_Year']+1\n",
    "\n",
    "\n",
    "# # Normalize Item_Visibility\n",
    "# scaler = MinMaxScaler()\n",
    "# test_data['Normalized_Visibility'] = scaler.fit_transform(test_data[['Item_Visibility']])\n",
    "\n",
    "# # Encode Outlet_Size\n",
    "# encoder = LabelEncoder()\n",
    "# test_data['Encoded_Outlet_Size'] = encoder.fit_transform(test_data['Outlet_Size'])\n",
    "# test_data['visibility_outlet'] = test_data['Normalized_Visibility'] + test_data['Encoded_Outlet_Size']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "# from pandas.plotting import parallel_coordinates\n",
    "# # Select columns for parallel coordinates plot\n",
    "# # selected_columns = ['Item_Outlet_Sales', 'Item_Weight', 'Item_Visibility']\n",
    "# selected_columns = ['Item_Weight', 'Item_Fat_Content', 'Item_Visibility', 'Item_Type', 'Item_MRP', 'outlet_rel_age', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type','Item_Outlet_Sales']\n",
    "\n",
    "\n",
    "# # Encode categorical columns\n",
    "# encoder = LabelEncoder()\n",
    "# for col in ['Item_Type', 'Outlet_Size','Item_Fat_Content','Outlet_Location_Type', 'Outlet_Type']:\n",
    "#     data[col] = encoder.fit_transform(data[col])\n",
    "\n",
    "# # Normalize the data\n",
    "# scaler = MinMaxScaler()\n",
    "# data[selected_columns] = scaler.fit_transform(data[selected_columns])\n",
    "\n",
    "# # Create a dummy class column\n",
    "# data['Dummy_Class'] = 'All'\n",
    "\n",
    "# # Plot parallel coordinates\n",
    "# plt.figure(figsize=(16, 4))\n",
    "# parallel_coordinates(data[selected_columns + ['Outlet_Identifier']], 'Outlet_Identifier', colormap='viridis')\n",
    "# plt.title('Parallel Coordinates Plot')\n",
    "# plt.xlabel('Features')\n",
    "# plt.ylabel('Values')\n",
    "# plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/mh/_6jfhlk17_v8nmgct4j_mlnh0000gn/T/ipykernel_32934/1081095424.py:16: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.\n",
      "The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.\n",
      "\n",
      "For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.\n",
      "\n",
      "\n",
      "  test_data['Item_MRP_Std_calculated'].fillna(0, inplace=True) #use the new column name.\n",
      "/var/folders/mh/_6jfhlk17_v8nmgct4j_mlnh0000gn/T/ipykernel_32934/1081095424.py:28: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.\n",
      "The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.\n",
      "\n",
      "For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.\n",
      "\n",
      "\n",
      "  data[col].fillna(mean_val, inplace=True)\n",
      "/var/folders/mh/_6jfhlk17_v8nmgct4j_mlnh0000gn/T/ipykernel_32934/1081095424.py:30: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.\n",
      "The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.\n",
      "\n",
      "For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.\n",
      "\n",
      "\n",
      "  test_data[col].fillna(mean_val, inplace=True)\n",
      "/var/folders/mh/_6jfhlk17_v8nmgct4j_mlnh0000gn/T/ipykernel_32934/1081095424.py:34: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.\n",
      "The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.\n",
      "\n",
      "For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.\n",
      "\n",
      "\n",
      "  data[col].fillna(mode_val, inplace=True)\n",
      "/var/folders/mh/_6jfhlk17_v8nmgct4j_mlnh0000gn/T/ipykernel_32934/1081095424.py:36: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.\n",
      "The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.\n",
      "\n",
      "For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.\n",
      "\n",
      "\n",
      "  test_data[col].fillna(mode_val, inplace=True)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Validation RMSE: 1035.9485400263068\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.metrics import mean_squared_error\n",
    "import xgboost as xgb\n",
    "\n",
    "\n",
    "# 2. Feature Engineering: Item_MRP_Std\n",
    "item_mrp_std = data.groupby('Item_Identifier')['Item_MRP'].std().reset_index()\n",
    "item_mrp_std.rename(columns={'Item_MRP': 'Item_MRP_Std_calculated'}, inplace=True) #rename the column.\n",
    "\n",
    "data = pd.merge(data, item_mrp_std, on='Item_Identifier', how='left')\n",
    "test_data = pd.merge(test_data, item_mrp_std, on='Item_Identifier', how='left')\n",
    "test_data['Item_MRP_Std_calculated'].fillna(0, inplace=True) #use the new column name.\n",
    "\n",
    "# 3. Basic Feature Engineering\n",
    "data['Outlet_Age'] = 2023 - data['Outlet_Establishment_Year']\n",
    "test_data['Outlet_Age'] = 2023 - test_data['Outlet_Establishment_Year']\n",
    "\n",
    "# 4. Handle Missing Values\n",
    "numerical_cols = data.select_dtypes(include=['number']).columns\n",
    "categorical_cols = data.select_dtypes(include=['object']).columns\n",
    "\n",
    "for col in numerical_cols:\n",
    "    mean_val = data[col].mean()\n",
    "    data[col].fillna(mean_val, inplace=True)\n",
    "    if col in test_data.columns:\n",
    "        test_data[col].fillna(mean_val, inplace=True)\n",
    "\n",
    "for col in categorical_cols:\n",
    "    mode_val = data[col].mode()[0]\n",
    "    data[col].fillna(mode_val, inplace=True)\n",
    "    if col in test_data.columns:\n",
    "        test_data[col].fillna(mode_val, inplace=True)\n",
    "\n",
    "# 5. Feature Selection\n",
    "selected_features = ['Item_MRP', 'Outlet_Type', 'Outlet_Age', 'Item_Visibility', 'Item_MRP_Std_calculated', 'Item_Type']\n",
    "\n",
    "# Prepare Data\n",
    "X = data[selected_features]\n",
    "y = data['Item_Outlet_Sales']\n",
    "X_test = test_data[selected_features]\n",
    "\n",
    "# Preprocessing\n",
    "numerical_features = X.select_dtypes(include=['number']).columns\n",
    "categorical_features = X.select_dtypes(include=['object']).columns\n",
    "\n",
    "numerical_transformer = StandardScaler()\n",
    "categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)\n",
    "\n",
    "preprocessor = ColumnTransformer(\n",
    "    transformers=[\n",
    "        ('num', numerical_transformer, numerical_features),\n",
    "        ('cat', categorical_transformer, categorical_features)\n",
    "    ])\n",
    "\n",
    "X_processed = preprocessor.fit_transform(X)\n",
    "X_test_processed = preprocessor.transform(X_test)\n",
    "\n",
    "# Train-Validation Split\n",
    "X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.1, random_state=42)\n",
    "\n",
    "# XGBoost Model\n",
    "model = xgb.XGBRegressor(\n",
    "    objective='reg:squarederror',\n",
    "    n_estimators=500,\n",
    "    learning_rate=0.01,\n",
    "    random_state=42\n",
    ")\n",
    "\n",
    "model.fit(\n",
    "    X_train, y_train,\n",
    "    eval_set=[(X_val, y_val)],\n",
    "    verbose=False\n",
    ")\n",
    "\n",
    "# Evaluate Model\n",
    "val_predictions = model.predict(X_val)\n",
    "val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))\n",
    "print(f\"Validation RMSE: {val_rmse}\")\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Predictions saved to submission.csv\n"
     ]
    }
   ],
   "source": [
    "# Predictions for Test Data\n",
    "predictions = model.predict(X_test_processed)\n",
    "\n",
    "# Create Submission File\n",
    "submission = pd.DataFrame({'Item_Identifier': test_data['Item_Identifier'],\n",
    "                           'Outlet_Identifier': test_data['Outlet_Identifier'],\n",
    "                           'Item_Outlet_Sales': predictions})\n",
    "\n",
    "submission.to_csv('submission.csv', index=False)\n",
    "print('Predictions saved to submission.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.21"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

import pandas as pd

# 1. Load the CSV file
file_path = 'product_data.csv'
df = pd.read_csv(file_path)

# 2. Check for duplicate rows
duplicate_rows = df[df.duplicated()]

# 3. Check for any erroneous data (e.g., negative or zero prices, negative stock levels, unrealistic warranty periods)
erroneous_data = {
    "Negative or Zero Prices": df[df['UnitPrice'] <= 0],
    "Negative Stock Levels": df[df['StockLevel'] < 0],
    "Unrealistic Warranty Periods": df[df['WarrantyPeriod'] < 0]
}

# 4. Standardize text data: stripping whitespace and converting to lowercase for text fields
df['ProductName'] = df['ProductName'].str.strip().str.lower()
df['Description'] = df['Description'].str.strip().str.lower()
df['Manufacturer'] = df['Manufacturer'].str.strip().str.lower()

# 5. Display results
len(duplicate_rows), erroneous_data, df.head()


# Visual 
import matplotlib.pyplot as plt
import seaborn as sns

# Set the theme for seaborn
sns.set_theme(style="whitegrid")

# 1. Biểu đồ cột thể hiện số lượng sản phẩm trong từng nhóm sản phẩm
plt.figure(figsize=(10, 6))
sns.countplot(y="ProductGroupID", data=df, order=df['ProductGroupID'].value_counts().index)
plt.title('Số lượng sản phẩm trong từng nhóm sản phẩm')
plt.xlabel('Số lượng sản phẩm')
plt.ylabel('ID nhóm sản phẩm')
plt.show()

# 2. Biểu đồ phân phối giá sản phẩm
plt.figure(figsize=(10, 6))
sns.histplot(df['UnitPrice'], bins=20, kde=True)
plt.title('Phân phối giá sản phẩm')
plt.xlabel('Giá sản phẩm')
plt.ylabel('Số lượng sản phẩm')
plt.show()

# 3. Biểu đồ đường cho mức tồn kho của sản phẩm
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x=df.index, y="StockLevel")
plt.title('Mức tồn kho của sản phẩm')
plt.xlabel('Index sản phẩm')
plt.ylabel('Mức tồn kho')
plt.show()



# 3.3.1
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = 'your_file_path_here.csv'
data = pd.read_csv(file_path)

# Create a 'Sales' column as UnitPrice * StockLevel
data['Sales'] = data['UnitPrice'] * data['StockLevel']

# Select features for the model (e.g., WarrantyPeriod, UnitPrice, StockLevel)
features = ['WarrantyPeriod', 'UnitPrice', 'StockLevel']
X = data[features]
y = data['Sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared Score:", r2)

# 3.3.2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = 'your_file_path_here.csv'
data = pd.read_csv(file_path)

# Create a 'Sales' column as UnitPrice * StockLevel
data['Sales'] = data['UnitPrice'] * data['StockLevel']

# Select features for the model (e.g., WarrantyPeriod, UnitPrice, StockLevel)
features = ['WarrantyPeriod', 'UnitPrice', 'StockLevel']
X = data[features]
y = data['Sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model using sklearn
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model using sklearn metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared Score:", r2)

# 3.3.3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
file_path = 'your_file_path_here.csv'
data = pd.read_csv(file_path)

# Create a 'Sales' column as UnitPrice * StockLevel
data['Sales'] = data['UnitPrice'] * data['StockLevel']

# Select features for the model (e.g., WarrantyPeriod, UnitPrice, StockLevel)
features = ['WarrantyPeriod', 'UnitPrice', 'StockLevel']
X = data[features]
y = data['Sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict sales for the test set (future sales)
future_sales_predictions = model.predict(X_test)

# Combine predictions with the original test data for comparison
test_results = pd.DataFrame({
    'WarrantyPeriod': X_test['WarrantyPeriod'],
    'UnitPrice': X_test['UnitPrice'],
    'StockLevel': X_test['StockLevel'],
    'Actual Sales': y_test,
    'Predicted Sales': future_sales_predictions
})

# Print the test results
print(test_results)




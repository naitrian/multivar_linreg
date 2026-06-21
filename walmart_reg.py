import numpy as np
import pandas as pd

df = pd.read_csv("Walmart_Sales.csv")
df_scaled = pd.DataFrame()

# Feature Engineering: Adding New Columns: From Date -> Year, Month, Day. From Temperature -> Is_Freezing and Is_Burning (Both as 1/0)

df['Is_Freezing'] = (df['Temperature'] < 32).astype(int)
df['Is_Burning']  = (df['Temperature'] > 80).astype(int)
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df=df.drop(columns="Date")


# Seperating target and features in pandas

y = df["Weekly_Sales"].values
df = df.drop(columns="Weekly_Sales")

# Seperating train and test data (80, 20) split

split_num = int(len(df) * 0.8)
x_train = df.iloc[:split_num]
X_test_raw = df.iloc[split_num:]

y_train = y[:split_num]
y_test = y[split_num:] 

# Feature Scaling: Making sure all features are in the right and similar range expect Binary ones obviously, going for Z-score normalization i.e. standardization

std_store = np.std(x_train["Store"].values)
std_temp = np.std(x_train["Temperature"].values)
std_fuel = np.std(x_train["Fuel_Price"].values)
std_cpi = np.std(x_train["CPI"].values)
std_unemp = np.std(x_train["Unemployment"].values)
std_year = np.std(x_train["Year"].values)
std_month = np.std(x_train["Month"].values)
std_day = np.std(x_train["Day"].values)

mean_store = np.mean(x_train['Store'].values)
mean_temp = np.mean(x_train['Temperature'].values)
mean_fuel = np.mean(x_train['Fuel_Price'].values)
mean_cpi = np.mean(x_train['CPI'].values)
mean_unemp = np.mean(x_train['Unemployment'].values)
mean_year = np.mean(x_train['Year'].values)
mean_month = np.mean(x_train['Month'].values)
mean_day = np.mean(x_train['Day'].values)


df_scaled["Store_Scaled"] = ( x_train["Store"] - mean_store ) / std_store
df_scaled["Temp_Scaled"] = ( x_train["Temperature"] - mean_temp ) / std_temp
df_scaled["Fuel_Scaled"] = ( x_train["Fuel_Price"] - mean_fuel ) / std_fuel
df_scaled["Cpi_Scaled"] = ( x_train["CPI"] - mean_cpi ) / std_cpi
df_scaled["Unemp_Scaled"] = ( x_train["Unemployment"] - mean_unemp ) / std_unemp
df_scaled["Year_Scaled"] = ( x_train["Year"] - mean_year ) / std_year
df_scaled["Month_Scaled"] = ( x_train["Month"] - mean_month ) / std_month
df_scaled["Day_Scaled"] = ( x_train["Day"] - mean_day ) / std_day
df_scaled["Freeze_Scaled"] = x_train["Is_Freezing"]
df_scaled["Burn_Scaled"] = x_train["Is_Burning"]
df_scaled["Holiday_Scaled"] = x_train["Holiday_Flag"]

# Setting test dataframe

df_test_scaled = pd.DataFrame()
df_test_scaled["Store_Scaled"] = (X_test_raw["Store"] - mean_store) / std_store
df_test_scaled["Temp_Scaled"] = (X_test_raw["Temperature"] - mean_temp) / std_temp
df_test_scaled["Fuel_Scaled"] = (X_test_raw["Fuel_Price"] - mean_fuel) / std_fuel
df_test_scaled["Cpi_Scaled"] = (X_test_raw["CPI"] - mean_cpi) / std_cpi
df_test_scaled["Unemp_Scaled"] = (X_test_raw["Unemployment"] - mean_unemp) / std_unemp
df_test_scaled["Year_Scaled"] = (X_test_raw["Year"] - mean_year) / std_year
df_test_scaled["Month_Scaled"] = (X_test_raw["Month"] - mean_month) / std_month
df_test_scaled["Day_Scaled"] = (X_test_raw["Day"] - mean_day) / std_day
df_test_scaled["Freeze_Scaled"] = X_test_raw["Is_Freezing"]
df_test_scaled["Burn_Scaled"] = X_test_raw["Is_Burning"]
df_test_scaled["Holiday_Scaled"] = X_test_raw["Holiday_Flag"]

# Iniializing training set in Numpy

x_train = df_scaled.values
x_test = df_test_scaled.values


# Setting weights baises and cost function

m = len(x_train)

w = np.zeros(11)
b = 0
iterations = 10000
learning_rate = 0.01

# Implementing Gradient Descent:

cost_history = [] 
for i in range(iterations+1):
 
 pred = np.dot(x_train,w) + b
 cost = 1/(2*m) * np.sum((pred - y_train)**2)
 error = pred - y_train

 cost_history.append(cost)

 w_temp = w - (learning_rate/m) * (np.dot(x_train.T, error))
 b_temp = b - ((learning_rate/m)) * np.sum(error)

 w = w_temp
 b = b_temp

 if i % 100 == 0:
  print(f"COST IS : {cost}")

print("FINAL WEIGHTS AND BAIS")
print(w)
print(b)



# FINAL WEIGHTS AND BAIS AFTER 100,000 training cycles:

# FINAL WEIGHTS AND BAIS
# TEMP: -18634.91224516   
# FUEL PRICE: 20675.70281511 
# CPI: -117248.75389963  
# UNEMP: -71860.33486827
# YEAR:  -34148.08931403   
# MONTH: 38042.14471798  
# DAY: -11747.56669703 
# FREEZE: -133428.63521807
# HEAT: -39270.42998527  
# HOLIDAY:  85034.80628138]
# BIAS: 1129251.0963336367

# Note: IS HIGHLY INACCURATE DUE TO SOME REASONS NEITHER I NOR EVERY LLM CAN FIND. THE MATH IS GOOD (ALTHOUGH THE NORMALIZATION METHOD IS SHIT. BUT TO BE FAIR ITS MY FIRST TIME OKAY)

# ALSO: FINAL COST WAS 148 BILLION ToT
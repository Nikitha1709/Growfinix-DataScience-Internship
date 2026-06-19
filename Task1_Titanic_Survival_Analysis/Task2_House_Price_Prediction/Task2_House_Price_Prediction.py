# ==========================================
# TASK 2: HOUSE PRICE PREDICTION
# Growfinix Internship Project
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import joblib

print("=================================")
print("HOUSE PRICE PREDICTION")
print("=================================")

# ----------------------------------
# SAMPLE DATASET
# ----------------------------------

data = {
    'area': [1000,1200,1500,1800,2000,2200,2500,2800,3000,3500],
    'bedrooms': [2,2,3,3,3,4,4,4,5,5],
    'price': [3000000,3500000,4500000,5000000,5500000,
              6000000,7000000,7500000,8000000,9000000]
}

df = pd.DataFrame(data)

print("\nDataset:")
print(df)

# ----------------------------------
# CHECK NULL VALUES
# ----------------------------------

print("\nMissing Values:")
print(df.isnull().sum())

# ----------------------------------
# VISUALIZATION
# ----------------------------------

plt.figure(figsize=(6,4))
plt.scatter(df['area'], df['price'])
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Price vs Area")
plt.savefig("price_vs_area.png")
plt.close()

plt.figure(figsize=(6,4))
plt.scatter(df['bedrooms'], df['price'])
plt.xlabel("Bedrooms")
plt.ylabel("Price")
plt.title("Price vs Bedrooms")
plt.savefig("price_vs_bedrooms.png")
plt.close()

# ----------------------------------
# SPLIT DATA
# ----------------------------------

X = df[['area', 'bedrooms']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ----------------------------------
# TRAIN MODEL
# ----------------------------------

model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel Trained Successfully")

# ----------------------------------
# PREDICTION
# ----------------------------------

predictions = model.predict(X_test)

result = pd.DataFrame({
    'Actual Price': y_test,
    'Predicted Price': predictions
})

print("\nPrediction Results:")
print(result)

result.to_csv(
    "prediction_results.csv",
    index=False
)

# ----------------------------------
# EVALUATION
# ----------------------------------

mae = mean_absolute_error(y_test, predictions)

mse = mean_squared_error(y_test, predictions)

r2 = r2_score(y_test, predictions)

print("\nModel Evaluation")
print("MAE :", mae)
print("MSE :", mse)
print("R2 Score :", r2)

# ----------------------------------
# REGRESSION LINE
# ----------------------------------

plt.figure(figsize=(6,4))
plt.scatter(y_test, predictions)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")

plt.title("Actual vs Predicted")

plt.savefig("actual_vs_predicted.png")
plt.close()

# ----------------------------------
# RESIDUAL PLOT
# ----------------------------------

residuals = y_test - predictions

plt.figure(figsize=(6,4))
plt.scatter(predictions, residuals)

plt.axhline(y=0)

plt.xlabel("Predicted Price")
plt.ylabel("Residuals")

plt.title("Residual Plot")

plt.savefig("residual_plot.png")
plt.close()

# ----------------------------------
# SAVE MODEL
# ----------------------------------

joblib.dump(
    model,
    "house_price_model.pkl"
)

print("\nModel Saved Successfully")

# ----------------------------------
# CUSTOM PREDICTION FUNCTION
# ----------------------------------

def predict_house_price(area, bedrooms):

    value = model.predict(
        [[area, bedrooms]]
    )

    return value[0]

# Example

new_price = predict_house_price(
    2400,
    4
)

print("\nPredicted Price for")
print("Area = 2400")
print("Bedrooms = 4")

print("Price =", round(new_price))

print("\nProject Completed Successfully")

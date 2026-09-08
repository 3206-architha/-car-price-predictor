import streamlit as st
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(page_title="Used Car Price Predictor", page_icon="🚗")
st.title("🚗 Used Car Price Predictor")
st.write("Predicts a used car's selling price — choose which ML model does the prediction.")

# ---------- Load data ----------
df = pd.read_csv("used_car_price_dataset.csv")

categorical_cols = ["brand", "fuel_type", "transmission"]
numeric_cols = ["car_age", "km_driven", "owners"]

X = df[categorical_cols + numeric_cols]
y = df["selling_price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------- Preprocessing: OneHotEncoder + StandardScaler inside a ColumnTransformer ----------
# Different tool from LabelEncoder: OneHotEncoder avoids implying a false order between
# categories (e.g. it won't treat "Diesel" as mathematically greater than "Petrol"),
# and StandardScaler puts numeric columns on a comparable scale.
preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", StandardScaler(), numeric_cols),
])

# ---------- Model choice ----------
model_options = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, max_depth=3, random_state=42),
}

st.subheader("Choose a Model")
model_name = st.selectbox("Prediction Model", list(model_options.keys()))

# Pipeline chains preprocessing + model into a single object, so raw input goes
# in one end and a prediction comes out the other — no manual encoding step needed.
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model_options[model_name]),
])
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

with st.expander("Model performance"):
    st.write(f"Model: {model_name}")
    st.write(f"R² score: {r2:.2f}")
    st.write(f"Mean absolute error: ₹{mae:,.0f}")

    st.write("Compare all models:")
    comparison = []
    for name, m in model_options.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", m)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        comparison.append({
            "Model": name,
            "R² score": round(r2_score(y_test, pred), 2),
            "MAE (₹)": round(mean_absolute_error(y_test, pred), 0)
        })
    st.dataframe(pd.DataFrame(comparison))

st.subheader("Enter Car Details")

col1, col2 = st.columns(2)
with col1:
    brand = st.selectbox("Brand", sorted(df["brand"].unique()))
    fuel_type = st.selectbox("Fuel Type", sorted(df["fuel_type"].unique()))
    transmission = st.selectbox("Transmission", sorted(df["transmission"].unique()))

with col2:
    car_age = st.slider("Car Age (years)", 0, 17, 5)
    km_driven = st.slider("Kilometers Driven", 0, 200000, 40000, step=1000)
    owners = st.slider("Number of Previous Owners", 1, 4, 1)

if st.button("Predict Selling Price"):
    input_data = pd.DataFrame([[brand, fuel_type, transmission, car_age, km_driven, owners]],
                               columns=categorical_cols + numeric_cols)

    prediction = pipeline.predict(input_data)[0]
    st.success(f"Predicted Selling Price ({model_name}): ₹{prediction:,.0f}")

    st.bar_chart(pd.DataFrame({
        "Value": [prediction, df["selling_price"].mean()]
    }, index=["Predicted", "Average Car"]))

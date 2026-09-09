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
st.write(
    "Predicts a used car's selling price (in ₹ Lakhs), trained on real CarDekho "
    "used-car listings — choose which ML model does the prediction."
)

# ---------- Load real data ----------
df = pd.read_csv("real_car_data.csv")

categorical_cols = ["Fuel_Type", "Seller_Type", "Transmission"]
numeric_cols = ["Present_Price", "Kms_Driven", "Owner", "Year"]

X = df[categorical_cols + numeric_cols]
y = df["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------- Preprocessing: OneHotEncoder + StandardScaler inside a ColumnTransformer ----------
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
    st.write(f"Mean absolute error: ₹{mae:.2f} Lakhs")

    st.write("Compare all models:")
    comparison = []
    for name, m in model_options.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", m)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        comparison.append({
            "Model": name,
            "R² score": round(r2_score(y_test, pred), 2),
            "MAE (₹ Lakhs)": round(mean_absolute_error(y_test, pred), 2)
        })
    st.dataframe(pd.DataFrame(comparison))

st.subheader("Enter Car Details")

col1, col2 = st.columns(2)
with col1:
    fuel_type = st.selectbox("Fuel Type", sorted(df["Fuel_Type"].unique()))
    seller_type = st.selectbox("Seller Type", sorted(df["Seller_Type"].unique()))
    transmission = st.selectbox("Transmission", sorted(df["Transmission"].unique()))

with col2:
    present_price = st.slider("Present (Ex-Showroom) Price (₹ Lakhs)", 0.5, 35.0, 6.0, step=0.1)
    kms_driven = st.slider("Kilometers Driven", 0, 200000, 40000, step=1000)
    owner = st.selectbox("Number of Previous Owners", sorted(df["Owner"].unique()))
    year = st.slider("Year of Purchase", 2003, 2018, 2015)

if st.button("Predict Selling Price"):
    input_data = pd.DataFrame([[fuel_type, seller_type, transmission, present_price, kms_driven, owner, year]],
                               columns=categorical_cols + numeric_cols)

    prediction = pipeline.predict(input_data)[0]
    st.success(f"Predicted Selling Price ({model_name}): ₹{prediction:.2f} Lakhs")

    st.bar_chart(pd.DataFrame({
        "Value": [prediction, df["Selling_Price"].mean()]
    }, index=["Predicted", "Average Car"]))

    st.subheader("Similar Real Listings")
    st.caption("Actual cars from the dataset with a similar fuel type, transmission and price range.")
    similar = df[
        (df["Fuel_Type"] == fuel_type) &
        (df["Transmission"] == transmission) &
        (df["Selling_Price"].between(prediction * 0.6, prediction * 1.4))
    ][["Car_Name", "Year", "Selling_Price", "Kms_Driven", "Fuel_Type", "Transmission", "Owner"]]

    if not similar.empty:
        st.dataframe(similar.reset_index(drop=True))
    else:
        st.write("No close matches found in the dataset for this combination.")

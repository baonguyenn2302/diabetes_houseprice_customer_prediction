# ==========================================
# WEB CLIENT FOR INTELLIGENT SYSTEMS (STREAMLIT)
# ==========================================
import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Intelligent Systems Suite",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Intelligent Systems Development Suite")
st.markdown("From Data Representation to Deployable Intelligent System")

# Điều hướng giữa 3 ứng dụng
app_choice = st.sidebar.selectbox(
    "Select Application",
    ["1. Diabetes Prediction", "2. House Price Prediction", "3. E-commerce Customer Behavior"]
)

# ------------------------------------------
# 1. DIABETES PREDICTION UI
# ------------------------------------------
if app_choice == "1. Diabetes Prediction":
    st.header("🩺 Application 1: Diabetes Prediction")
    st.write("Enter patient clinical attributes below to assess diabetes risk.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        times_pregnant = st.number_input("Times Pregnant", min_value=0, max_value=20, value=2)
        glucose_concentration = st.number_input("Glucose Concentration", min_value=0.0, max_value=300.0, value=148.0)
        diastolic_blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=0.0, max_value=200.0, value=72.0)
        
    with col2:
        triceps_sf_thickness = st.number_input("Triceps Skin Fold (mm)", min_value=0.0, max_value=100.0, value=35.0)
        serum_insulin = st.number_input("Serum Insulin (mu U/ml)", min_value=0.0, max_value=900.0, value=0.0)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=33.6)
        
    with col3:
        d_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.627)
        years_of_age = st.number_input("Age (years)", min_value=1, max_value=120, value=50)

    if st.button("Predict Diabetes Risk", type="primary"):
        payload = {
            "times_pregnant": times_pregnant,
            "glucose_concentration": glucose_concentration,
            "diastolic_blood_pressure": diastolic_blood_pressure,
            "triceps_sf_thickness": triceps_sf_thickness,
            "serum_insulin": serum_insulin,
            "bmi": bmi,
            "d_pedigree_function": d_pedigree_function,
            "years_of_age": years_of_age
        }
        
        try:
            response = requests.post(f"{API_URL}/predict/diabetes", json=payload)
            if response.status_code == 200:
                data = response.json()
                st.success(f"**Prediction:** {data['prediction']}")
                st.info(f"**Model Confidence:** {data['confidence'] * 100:.2f}%")
            else:
                st.error("API Request Error")
        except Exception as e:
            st.error(f"Cannot connect to API Server: {e}")

# ------------------------------------------
# 2. HOUSE PRICE PREDICTION UI
# ------------------------------------------
elif app_choice == "2. House Price Prediction":
    st.header("🏠 Application 2: House Price Prediction")
    st.write("Enter property characteristics to estimate market value.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gr_liv_area = st.number_input("Above Grade Living Area (sq ft)", min_value=100.0, max_value=10000.0, value=1710.0)
        bedroom_abv_gr = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
        full_bath = st.number_input("Full Bathrooms", min_value=1, max_value=5, value=2)
        year_built = st.number_input("Year Built", min_value=1800, max_value=2026, value=2003)
        
    with col2:
        garage_cars = st.number_input("Garage Capacity (Cars)", min_value=0, max_value=5, value=2)
        lot_area = st.number_input("Lot Size (sq ft)", min_value=500.0, max_value=100000.0, value=8450.0)
        ms_zoning = st.selectbox("Zoning Classification", ["RL", "RM", "C (all)", "FV", "RH"])
        neighborhood = st.selectbox("Neighborhood", ["CollgCr", "Veenker", "Crawfor", "NoRidge", "Mitchel", "Somerst"])
        house_style = st.selectbox("House Style", ["2Story", "1Story", "1.5Fin", "SLvl"])

    if st.button("Estimate House Price", type="primary"):
        payload = {
            "GrLivArea": gr_liv_area,
            "BedroomAbvGr": bedroom_abv_gr,
            "FullBath": full_bath,
            "YearBuilt": year_built,
            "GarageCars": garage_cars,
            "LotArea": lot_area,
            "MSZoning": ms_zoning,
            "Neighborhood": neighborhood,
            "HouseStyle": house_style
        }
        
        try:
            response = requests.post(f"{API_URL}/predict/house-price", json=payload)
            if response.status_code == 200:
                data = response.json()
                st.success(f"**Predicted House Price:** ${data['predicted_price']:,.2f} {data['currency']}")
            else:
                st.error("API Request Error")
        except Exception as e:
            st.error(f"Cannot connect to API Server: {e}")

# ------------------------------------------
# 3. E-COMMERCE CUSTOMER BEHAVIOR UI
# ------------------------------------------
else:
    st.header("🛒 Application 3: E-commerce Customer Behavior Discovery")
    st.write("Analyze customer interaction and review text to discover product interest.")
    
    num_helpful = st.number_input("Helpful Votes on Review", min_value=0, max_value=100, value=2)
    customer_review = st.text_area(
        "Customer Review Text",
        value="This tablet has an amazing display and fast performance. Battery life lasts for days."
    )

    if st.button("Discover Customer Interest", type="primary"):
        payload = {
            "num_helpful": num_helpful,
            "customer_review": customer_review
        }
        
        try:
            response = requests.post(f"{API_URL}/predict/customer-behavior", json=payload)
            if response.status_code == 200:
                data = response.json()
                st.success(f"**Predicted Customer Interest:** {data['predicted_interest']}")
                st.info(f"**Confidence:** {data['confidence'] * 100:.2f}%")
            else:
                st.error("API Request Error")
        except Exception as e:
            st.error(f"Cannot connect to API Server: {e}")
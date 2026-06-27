import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Customer Churn Risk Prediction System",
    page_icon="🔮",
    layout="wide"
)

# Custom SaaS Dashboard Design CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;700&display=swap');

/* Global colors and body overrides */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: #0B1120 !important;
    color: #E2E8F0 !important;
    font-family: 'Inter', sans-serif !important;
}

/* Top header background blur */
[data-testid="stHeader"] {
    background-color: rgba(11, 17, 32, 0.8) !important;
    backdrop-filter: blur(10px) !important;
}

/* Headings typography */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #E2E8F0 !important;
    font-weight: 700 !important;
}

/* Sidebar background styling */
[data-testid="stSidebar"], [data-testid="stSidebarNav"] {
    background-color: #0F1729 !important;
    border-right: 1px solid #1E293B !important;
}

/* Form input elements style override (dark surface #151D2E) */
input, div[data-baseweb="select"] > div, div[data-baseweb="input"] {
    background-color: #151D2E !important;
    color: #E2E8F0 !important;
    border-color: #1E293B !important;
    border-radius: 8px !important;
}

input:focus, div[data-baseweb="select"]:focus-within {
    border-color: #3DD9C4 !important;
    box-shadow: 0 0 0 1px #3DD9C4 !important;
}

/* Selectbox dropdown elements styling */
div[role="listbox"] {
    background-color: #151D2E !important;
    color: #E2E8F0 !important;
    border: 1px solid #1E293B !important;
}

div[role="option"] {
    background-color: #151D2E !important;
    color: #E2E8F0 !important;
}

div[role="option"]:hover, div[role="option"][aria-selected="true"] {
    background-color: #1E293B !important;
    color: #3DD9C4 !important;
}

/* Expander panel custom style (behaves like a widget card) */
[data-testid="stExpander"] {
    background-color: #151D2E !important;
    border: 1px solid #1E293B !important;
    border-radius: 8px !important;
    margin-bottom: 16px !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
}

[data-testid="stExpander"] details {
    border: none !important;
}

/* Primary predict button custom style (teal background) */
div[data-testid="stButton"] button[kind="primary"] {
    background-color: #3DD9C4 !important;
    color: #0B1120 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 24px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 10px rgba(61, 217, 196, 0.2) !important;
    width: 100% !important;
}

div[data-testid="stButton"] button[kind="primary"]:hover {
    background-color: #4dfce2 !important;
    box-shadow: 0 0 18px rgba(61, 217, 196, 0.5) !important;
    transform: translateY(-1px) !important;
}

/* Secondary reset button custom style (bordered) */
div[data-testid="stButton"] button[kind="secondary"] {
    background-color: transparent !important;
    color: #94A3B8 !important;
    border: 1px solid #1E293B !important;
    border-radius: 8px !important;
    padding: 12px 24px !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
}

div[data-testid="stButton"] button[kind="secondary"]:hover {
    color: #E2E8F0 !important;
    border-color: #94A3B8 !important;
}

/* Premium Churn Risk Card */
.risk-card {
    background-color: #151D2E;
    border-radius: 12px;
    padding: 28px;
    margin-bottom: 24px;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
}

.risk-card-low {
    border-left: 6px solid #34D399;
}

.risk-card-medium {
    border-left: 6px solid #FBBF24;
}

.risk-card-high {
    border-left: 6px solid #F87171;
}

.risk-percent {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 56px;
    font-weight: 700;
    line-height: 1.1;
    margin-bottom: 12px;
}

.risk-percent-low {
    color: #34D399;
}

.risk-percent-medium {
    color: #FBBF24;
}

.risk-percent-high {
    color: #F87171;
}

.risk-badge {
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.5px;
    font-family: 'Space Grotesk', sans-serif;
    border: 1px solid currentColor;
}

.risk-title {
    font-size: 20px;
    font-weight: 600;
    margin-top: 8px;
    font-family: 'Space Grotesk', sans-serif;
}

.risk-description {
    color: #94A3B8;
    font-size: 14px;
    line-height: 1.6;
    margin-top: 6px;
}

/* Key factors and action cards */
.custom-card {
    background-color: #151D2E;
    border: 1px solid #1E293B;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 20px;
}

/* Styled lists */
ul.recommendation-list {
    list-style-type: none;
    padding-left: 0;
    margin: 0;
}

ul.recommendation-list li {
    padding-left: 28px;
    position: relative;
    margin-bottom: 10px;
    font-size: 14px;
    line-height: 1.5;
}

ul.recommendation-list li::before {
    content: "⚡";
    position: absolute;
    left: 0;
    top: 0;
    color: #3DD9C4;
}

/* Table styling customization */
[data-testid="stTable"] table {
    background-color: #151D2E !important;
    color: #E2E8F0 !important;
    border-collapse: collapse !important;
    border-radius: 8px !important;
    overflow: hidden !important;
    border: 1px solid #1E293B !important;
    width: 100% !important;
}

[data-testid="stTable"] th {
    background-color: #0F1729 !important;
    color: #E2E8F0 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    border-bottom: 2px solid #1E293B !important;
    padding: 12px !important;
    font-weight: 600 !important;
}

[data-testid="stTable"] td {
    border-bottom: 1px solid #1E293B !important;
    padding: 10px 12px !important;
    color: #E2E8F0 !important;
}

/* Caption and footer text */
.stCaption, caption {
    color: #94A3B8 !important;
}

/* Notification styling */
div[data-testid="stNotification"] {
    background-color: #151D2E !important;
    border: 1px solid #1E293B !important;
    color: #E2E8F0 !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

# Define file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model.pkl')
ENCODERS_PATH = os.path.join(BASE_DIR, 'models', 'encoders.pkl')
FEATURE_NAMES_PATH = os.path.join(BASE_DIR, 'models', 'feature_names.pkl')

# Model, encoder, and feature names loading function (Cached for performance)
@st.cache_resource
def load_resources():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODERS_PATH) or not os.path.exists(FEATURE_NAMES_PATH):
        raise FileNotFoundError(
            "Required model and encoder files not found. Please make sure to run data preparation (data_prep.py) "
            "and training (train.py) steps first."
        )
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    return model, encoders, feature_names

# Default values dictionary
DEFAULTS = {
    'gender': "Female",
    'SeniorCitizen': "No",
    'Partner': "No",
    'Dependents': "No",
    'tenure': 12,
    'Contract': "Month-to-month",
    'PaperlessBilling': "Yes",
    'PaymentMethod': "Electronic check",
    'MonthlyCharges': 50.0,
    'TotalCharges': 600.0,
    'PhoneService': "Yes",
    'MultipleLines': "No",
    'InternetService': "Fiber optic",
    'OnlineSecurity': "No",
    'OnlineBackup': "No",
    'DeviceProtection': "No",
    'TechSupport': "No",
    'StreamingTV': "No",
    'StreamingMovies': "No",
    'prediction_results': None
}

# Initialize session state variables if not present
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

# Callback function to reset all inputs and results
def reset_inputs():
    for key, val in DEFAULTS.items():
        st.session_state[key] = val

# Load and verify resources
try:
    model, encoders, feature_names = load_resources()
    resources_ready = True
except Exception as e:
    st.error(f"⚠️ An error occurred while loading resources: {e}")
    st.info("Please verify that the model training completed successfully and model files were generated.")
    resources_ready = False

# Load best model metrics from comparison.csv
@st.cache_data
def get_best_model_metrics():
    comparison_path = os.path.join(BASE_DIR, 'models', 'comparison.csv')
    if os.path.exists(comparison_path):
        try:
            df_comp = pd.read_csv(comparison_path)
            best_idx = df_comp['ROC_AUC'].idxmax()
            best_row = df_comp.loc[best_idx]
            model_name = best_row['Model'].replace('_', ' ')
            roc_auc = best_row['ROC_AUC']
            return f"{model_name} &bull; ROC-AUC: {roc_auc:.3f}"
        except Exception:
            return "Logistic Regression &bull; ROC-AUC: 0.829"
    return "Logistic Regression &bull; ROC-AUC: 0.829"

# Render SaaS Header Bar
def render_header():
    best_model_badge = get_best_model_metrics()
    header_col1, header_col2 = st.columns([70, 30])
    with header_col1:
        st.title("🔮 Customer Churn Prediction Portal")
        st.markdown("<p style='color:#94A3B8; font-size:16px; margin-top:-10px;'>Analyze customer churn risks with machine learning and get actionable retention recommendations.</p>", unsafe_allow_html=True)
    with header_col2:
        st.markdown(f"""
        <div style='text-align: right; margin-top: 20px;'>
            <span style='background-color: #151D2E; border: 1px solid #1E293B; color: #3DD9C4; padding: 6px 14px; border-radius: 20px; font-family: "Space Grotesk", sans-serif; font-size: 13px; font-weight: 600; display: inline-flex; align-items: center;'>
                ⚡ {best_model_badge}
            </span>
        </div>
        """, unsafe_allow_html=True)

# Helper function to encode categorical values matching the LabelEncoders
def encode_value(column_name, value):
    if column_name in encoders:
        try:
            return encoders[column_name].transform([str(value)])[0]
        except Exception:
            return 0
    return value

# Function to get key risk factors based on inputs
def get_risk_factors(inputs_raw):
    factors = []
    if inputs_raw['Contract'] == "Month-to-month":
        factors.append("Short-Term Contract: Month-to-month plans show historically high churn rates.")
    if inputs_raw['tenure'] <= 6:
        factors.append("New Customer Cohort: Customers within their first 6 months are statistically highly vulnerable.")
    if inputs_raw['PaymentMethod'] == "Electronic check":
        factors.append("Payment Method: Manual electronic check payments have higher churn than automatic payment types.")
    if inputs_raw['InternetService'] == "Fiber optic":
        factors.append("Fiber Optic: Fiber optic subscribers show slightly higher churn tendencies in this customer cohort.")
    if inputs_raw['OnlineSecurity'] == "No":
        factors.append("Lacking Online Security: Accounts without security add-ons show lower switching costs.")
    if inputs_raw['TechSupport'] == "No":
        factors.append("No Tech Support: Lack of dedicated technical support lowers overall service satisfaction.")
    
    return factors[:3]

# Create interface layout
if resources_ready:
    render_header()
    
    # Divide the main dashboard area into two columns: 35% form inputs, 65% results
    left_col, right_col = st.columns([35, 65])
    
    # 2. LEFT COLUMN (Form inputs grouped logically)
    with left_col:
        st.markdown("<h3 style='margin-bottom:12px;'>📋 Customer Profile</h3>", unsafe_allow_html=True)
        
        with st.expander("👤 Demographics", expanded=True):
            gender = st.selectbox("Gender", ["Female", "Male"], key="gender")
            senior_citizen_str = st.selectbox("Senior Citizen", ["No", "Yes"], key="SeniorCitizen")
            senior_citizen = 1 if senior_citizen_str == "Yes" else 0
            partner = st.selectbox("Partner", ["No", "Yes"], key="Partner")
            dependents = st.selectbox("Dependents", ["No", "Yes"], key="Dependents")
            
        with st.expander("📄 Account Details", expanded=True):
            tenure = st.number_input("Customer Tenure (months)", min_value=0, max_value=72, key="tenure")
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], key="Contract")
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"], key="PaperlessBilling")
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ], key="PaymentMethod")
            
        with st.expander("🛠️ Services Subscribed", expanded=False):
            phone_service = st.selectbox("Phone Service", ["Yes", "No"], key="PhoneService")
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], key="MultipleLines")
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], key="InternetService")
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"], key="OnlineSecurity")
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"], key="OnlineBackup")
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"], key="DeviceProtection")
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], key="TechSupport")
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"], key="StreamingTV")
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"], key="StreamingMovies")
            
        with st.expander("💰 Billing Details", expanded=True):
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=150.0, step=0.1, key="MonthlyCharges")
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, step=1.0, key="TotalCharges")
            
        # 6. Input validation checks
        expected_charges = tenure * monthly_charges
        if tenure > 0 and abs(total_charges - expected_charges) / expected_charges > 0.5:
            st.warning(f"⚠️ Billing Inconsistency: Total charges (${total_charges:,.2f}) differ significantly from calculated charges (${expected_charges:,.2f}) based on tenure and monthly rates.")
            
        # Form buttons layout
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            predict_clicked = st.button("🔮 Predict Risk", type="primary", use_container_width=True)
        with btn_col2:
            reset_clicked = st.button("🔄 Reset Form", type="secondary", on_click=reset_inputs, use_container_width=True)
            
    # Process prediction if button is clicked
    if predict_clicked:
        # Calculate engineered features
        services_list = [phone_service, multiple_lines, online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies]
        num_services = sum(1 for s in services_list if s == 'Yes')
        avg_monthly_spend = monthly_charges if tenure == 0 else total_charges / tenure
        is_new_customer = 1 if tenure <= 6 else 0
        
        # Build features input vector
        encoded_data = {
            'gender': encode_value('gender', gender),
            'SeniorCitizen': senior_citizen,
            'Partner': encode_value('Partner', partner),
            'Dependents': encode_value('Dependents', dependents),
            'tenure': tenure,
            'PhoneService': encode_value('PhoneService', phone_service),
            'MultipleLines': encode_value('MultipleLines', multiple_lines),
            'InternetService': encode_value('InternetService', internet_service),
            'OnlineSecurity': encode_value('OnlineSecurity', online_security),
            'OnlineBackup': encode_value('OnlineBackup', online_backup),
            'DeviceProtection': encode_value('DeviceProtection', device_protection),
            'TechSupport': encode_value('TechSupport', tech_support),
            'StreamingTV': encode_value('StreamingTV', streaming_tv),
            'StreamingMovies': encode_value('StreamingMovies', streaming_movies),
            'Contract': encode_value('Contract', contract),
            'PaperlessBilling': encode_value('PaperlessBilling', paperless_billing),
            'PaymentMethod': encode_value('PaymentMethod', payment_method),
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges,
            'NumServices': num_services,
            'AvgMonthlySpend': avg_monthly_spend,
            'IsNewCustomer': is_new_customer
        }
        
        # Enforce exact column training order
        df_predict = pd.DataFrame([encoded_data])[feature_names]
        
        # Execute prediction probabilities
        churn_proba = model.predict_proba(df_predict)[0][1]
        
        # Save results in session state to persist on state updates
        st.session_state['prediction_results'] = {
            'proba': float(churn_proba),
            'inputs_raw': {
                'Contract': contract,
                'tenure': tenure,
                'PaymentMethod': payment_method,
                'InternetService': internet_service,
                'OnlineSecurity': online_security,
                'TechSupport': tech_support,
                'gender': gender,
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges,
                'num_services': num_services,
                'is_new_customer': is_new_customer
            }
        }
        
    # 2. RIGHT COLUMN (Results view or placeholder)
    with right_col:
        st.markdown("<h3 style='margin-bottom:12px;'>📊 Risk Evaluation</h3>", unsafe_allow_html=True)
        
        # Show placeholder if no prediction has been executed
        if st.session_state['prediction_results'] is None:
            st.markdown("""
            <div style='border: 2px dashed #1E293B; border-radius: 12px; padding: 80px 40px; text-align: center; background-color: #0F1729; margin-top: 20px;'>
                <div style='font-size: 64px; margin-bottom: 20px; color: #94A3B8;'>📊</div>
                <h3 style='color: #E2E8F0; margin-bottom: 8px; font-family: "Space Grotesk", sans-serif;'>Ready for Evaluation</h3>
                <p style='color: #94A3B8; max-width: 400px; margin: 0 auto; font-size: 14px;'>Configure the customer profile inputs on the left side and click <b>Predict Risk</b> to run the churn prediction model and view details.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Retrive prediction details from state
            results = st.session_state['prediction_results']
            proba_pct = results['proba'] * 100
            inputs_raw = results['inputs_raw']
            
            # Setup custom risk styling and details
            if proba_pct < 30.0:
                card_class = "risk-card-low"
                pct_class = "risk-percent-low"
                risk_tier = "LOW"
                risk_color = "#34D399"
                badge_bg = "rgba(52, 211, 153, 0.1)"
                risk_desc = "This customer shows stable behavior and is highly likely to continue their membership. Maintain standard service quality."
            elif proba_pct < 60.0:
                card_class = "risk-card-medium"
                pct_class = "risk-percent-medium"
                risk_tier = "MEDIUM"
                risk_color = "#FBBF24"
                badge_bg = "rgba(251, 191, 36, 0.1)"
                risk_desc = "The customer exhibits moderate churn tendencies. Monitor metrics closely and consider scheduling proactive loyalty campaigns."
            else:
                card_class = "risk-card-high"
                pct_class = "risk-percent-high"
                risk_tier = "HIGH"
                risk_color = "#F87171"
                badge_bg = "rgba(248, 113, 113, 0.1)"
                risk_desc = "Critical churn alert! This customer is highly likely to cancel services. Immediate retention outreach is recommended."
                
            # Render visual risk evaluation card
            st.markdown(f"""
            <div class="risk-card {card_class}">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span class="risk-title" style="margin: 0; color: #E2E8F0;">Churn Propensity Analysis</span>
                    <span class="risk-badge" style="background-color: {badge_bg}; color: {risk_color};">{risk_tier} RISK</span>
                </div>
                <div class="risk-percent {pct_class}">{proba_pct:.1f}%</div>
                <div class="risk-description">{risk_desc}</div>
                <div style="margin-top: 20px;">
                    <span style="font-size: 12px; color: #94A3B8; font-weight: 500;">Churn Probability Gauge</span>
                    <div style="background-color: #1E293B; border-radius: 8px; height: 12px; width: 100%; margin-top: 6px; overflow: hidden;">
                        <div style="background-color: {risk_color}; width: {proba_pct}%; height: 100%; border-radius: 8px;"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Key Risk Factors (Rule-based overview)
            st.markdown("#### 🔍 Key Risk Factors")
            factors = get_risk_factors(inputs_raw)
            
            if factors:
                factor_html = "<div class='custom-card'>"
                for factor in factors:
                    factor_html += f"<p style='margin: 0 0 10px 0; font-size: 14px; color: #E2E8F0;'>&bull; {factor}</p>"
                factor_html += "</div>"
                st.markdown(factor_html, unsafe_allow_html=True)
            else:
                st.markdown("<div class='custom-card'><p style='margin:0; font-size:14px; color:#94A3B8;'>No critical risk factors detected for this profile.</p></div>", unsafe_allow_html=True)
                
            # 3. Recommended Actions Section
            st.markdown("#### 💡 Recommended Actions")
            if proba_pct < 30.0:
                st.markdown("""
                <div class="custom-card">
                    <ul class="recommendation-list">
                        <li><b>Standard Customer Health Checks:</b> Schedule automatic quality checks during standard milestones.</li>
                        <li><b>Review Upselling Options:</b> Identify opportunities to introduce value-adding premium subscriptions.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            elif proba_pct < 60.0:
                st.markdown("""
                <div class="custom-card">
                    <ul class="recommendation-list">
                        <li><b>Automatic Payment Promotion:</b> Encourage transition from manual check billing to automatic bank or card drafts with a minor one-time discount.</li>
                        <li><b>Provide Service Value Bundle:</b> Introduce complementary trial periods for security or backups to raise customer retention.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="custom-card">
                    <ul class="recommendation-list">
                        <li><b>Urgent Term-Extension Offer:</b> The customer has a "{inputs_raw['Contract']}" plan. Present an immediate term discount to secure a 1 or 2-year contract duration.</li>
                        <li><b>Add Complementary Support Services:</b> Package free Tech Support or Online Security modules to resolve potential pain points.</li>
                        <li><b>Priority Retention Desk Outreach:</b> Assign this customer account to the personal retention staff for immediate follow-up.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
            # Compact Customer Profile Summary Table
            st.markdown("#### 📋 Profile Data Summary")
            summary_df = pd.DataFrame({
                "Parameter": [
                    "Gender", "Tenure (Months)", "Contract Type", "Internet Service", 
                    "Monthly Charges", "Total Charges", "Active Services", "New Customer Status"
                ],
                "Value": [
                    inputs_raw['gender'], f"{inputs_raw['tenure']} Months", inputs_raw['Contract'], inputs_raw['InternetService'], 
                    f"{inputs_raw['MonthlyCharges']:.2f} $", f"{inputs_raw['TotalCharges']:.2f} $", 
                    f"{inputs_raw['num_services']} Services", "Yes" if inputs_raw['is_new_customer'] else "No"
                ]
            })
            st.table(summary_df)

# Footer Area
st.markdown("---")
footer_col1, footer_col2 = st.columns([70, 30])
with footer_col1:
    st.caption("ℹ️ Model trained on Telco Customer Churn dataset. For demonstration and decision support purposes only.")
with footer_col2:
    st.markdown("<p style='text-align: right; font-size: 12px; margin: 0; color: #94A3B8;'>Code Repository: <a href='#' style='color:#3DD9C4; text-decoration:none;'>GitHub Project</a></p>", unsafe_allow_html=True)

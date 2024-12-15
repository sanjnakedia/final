import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import OrdinalEncoder
from framingham10yr import framingham_10year_risk  # Import the function
import pandas as pd
import time
import plotly.graph_objects as go


if 'current_view' not in st.session_state:
    st.session_state.current_view = None

# ... (keep all imports and initial setup)
st.header("South Asian Heart Disease Risk Calculator")
st.write("**Please fill out the questionnaire below to see if you are at risk of heart disease:**")
## Loading Data
#df = pd.read_csv('data/heart_2020_cleaned_with_synthetic.csv')
#newdf = df.copy()

#st.write("**Please fill out the questionnaire below to see if you are at risk of heart disease:**")

def plot_overlayed_histogram(normal_scores, sa_adjusted_scores, variable_label):
    """
    Creates an overlayed histogram for normal and South Asian-adjusted risk scores.
    """
    fig = go.Figure()

    # Add Normal Distribution
    fig.add_trace(go.Histogram(
        x=normal_scores,
        name='Normal Risk Scores',
        opacity=0.75,
        marker_color='blue'
    ))

    # Add SA-Adjusted Distribution
    fig.add_trace(go.Histogram(
        x=sa_adjusted_scores,
        name='South Asian Adjusted Risk Scores',
        opacity=0.75,
        marker_color='red'
    ))

    # Update layout
    fig.update_layout(
        barmode='overlay',
        title=f"Distribution of Risk Scores ({variable_label})",
        xaxis_title="Risk Score (%)",
        yaxis_title="Frequency",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white"
    )

    return fig

    
def compute_ten_year_score(metrics):
    try:
        # Extract metrics
        age = metrics.get('age', 50)
        gender = metrics.get('gender', 'male')
        cholesterol_tot = metrics.get('cholesterol_tot', 200)
        cholesterol_hdl = metrics.get('cholesterol_hdl', 50) 
        bp_systolic = metrics.get('bp_systolic', 120)
        diabetes_status = metrics.get('diabetes_status', 'no')
        smoking_status = metrics.get('smoking_status', 'never')
        hypertension_treatment = metrics.get('hypertension_treatment', 'no') 

        # Validations
        if int(age) < 20 or int(age) > 59:
            return None

        # Natural logs
        ln_age = math.log(age)
        ln_total_chol = math.log(cholesterol_tot)
        ln_hdl = math.log(cholesterol_hdl)
        ln_sbp = math.log(bp_systolic)

        trlnsbp = ln_sbp if hypertension_treatment == "yes" else 0
        ntlnsbp = ln_sbp if hypertension_treatment != "yes" else 0
        
        age_total_chol = ln_age * ln_total_chol
        age_hdl = ln_age * ln_hdl
        aget_sbp = ln_age * trlnsbp
        agent_sbp = ln_age * ntlnsbp
        age_smoke = ln_age if smoking_status == "current" else 0

        # Coefficients based on race and gender
        if gender == "female":
            s010_ret, mnxb_ret = 0.96652, -29.1817
            predict_ret = (
                -29.799 * ln_age + 4.884 * (ln_age ** 2) + 13.54 * ln_total_chol -
                3.114 * age_total_chol - 13.578 * ln_hdl + 3.149 * age_hdl +
                2.019 * trlnsbp + 1.957 * ntlnsbp +
                (7.574 if smoking_status == "current" else 0) -
                1.665 * age_smoke +
                (0.661 if diabetes_status == "yes" else 0)
            )
        else: 
            s010_ret, mnxb_ret = 0.91436, 61.1816
            predict_ret = (
                12.344 * ln_age + 11.853 * ln_total_chol - 2.664 * age_total_chol -
                7.99 * ln_hdl + 1.769 * age_hdl +
                1.797 * trlnsbp + 1.764 * ntlnsbp +
                (7.837 if smoking_status == "current" else 0) -
                1.795 * age_smoke +
                (0.658 if diabetes_status == "yes" else 0)
            )

        pct = 1 - math.pow(s010_ret, math.exp(predict_ret - mnxb_ret))
        return round(pct * 100, 1)

    except Exception as e:
        print(f"Error in compute_ten_year_score: {e}")
        return None  # Return None for invalid/missing input


def compute_SA_ten_year_score(metrics):
    # Extract metrics
    age = metrics['age']
    gender = metrics['gender']
    cholesterol_tot = metrics['cholesterol_tot']
    cholesterol_hdl = metrics['cholesterol_hdl']
    bp_systolic = metrics['bp_systolic']
    diabetes_status = metrics['diabetes_status']
    smoking_status = metrics['smoking_status']
    hypertension_treatment = metrics['hypertension_treatment']

    # Validations
    if age < 20 or age > 59:
        return None

    # Natural logs
    ln_age = math.log(age)
    ln_total_chol = math.log(cholesterol_tot)
    ln_hdl = math.log(cholesterol_hdl)
    ln_sbp = math.log(bp_systolic)

    trlnsbp = ln_sbp if hypertension_treatment == "yes" else 0
    ntlnsbp = ln_sbp if hypertension_treatment != "yes" else 0
    
    age_total_chol = ln_age * ln_total_chol
    age_hdl = ln_age * ln_hdl
    aget_sbp = ln_age * trlnsbp
    agent_sbp = ln_age * ntlnsbp
    age_smoke = ln_age if smoking_status == "current" else 0

    # Coefficients based on race and gender
    if gender == "female":
        s010_ret, mnxb_ret = 0.96652, -29.1817
        predict_ret =  (
            -29.799 * ln_age + 4.884 * (ln_age ** 2) + 13.54 * ln_total_chol -
            3.114 * age_total_chol - 13.578 * ln_hdl + 3.149 * age_hdl +
            2.019 * trlnsbp + 1.957 * ntlnsbp +
            (7.574 if smoking_status == "current" else 0) -
            1.665 * age_smoke +
            (0.661 if diabetes_status == "yes" else 0)
        )
    else: 
        s010_ret, mnxb_ret = 0.91436, 61.1816
        predict_ret = (
            12.344 * ln_age + 11.853 * ln_total_chol - 2.664 * age_total_chol -
            7.99 * ln_hdl + 1.769 * age_hdl +
            1.797 * trlnsbp + 1.764 * ntlnsbp +
            (7.837 if smoking_status == "current" else 0) -
            1.795 * age_smoke +
            (0.658 if diabetes_status == "yes" else 0)
        )

    pct = (1 - math.pow(s010_ret, math.exp(predict_ret - mnxb_ret))) * 1.5
    return round(pct * 100, 1)



def compute_lifetime_risk(metrics):
    """
    Compute the lifetime ASCVD risk score.
    
    Parameters:
        age (int): Age of the individual (20-59).
        gender (str): Gender ("male" or "female").
        total_cholesterol (int): Total cholesterol level (mg/dL).
        systolic_blood_pressure (int): Systolic blood pressure (mm Hg).
        hdl (int): HDL cholesterol level (mg/dL).
        diabetic (bool): Whether the individual has diabetes.
        smoker (bool): Whether the individual is a smoker.
        hypertensive (bool): Whether the individual is on hypertension treatment.
    
    Returns:
        int or None: Lifetime ASCVD risk score or None if age is out of range.
    """
    age = metrics['age']
    gender = metrics['gender']
    cholesterol_tot = metrics['cholesterol_tot']
    cholesterol_hdl = metrics['cholesterol_hdl']
    bp_systolic = metrics['bp_systolic']
    diabetes_status = metrics['diabetes_status']
    smoking_status = metrics['smoking_status']
    hypertension_treatment = metrics['hypertension_treatment']

    if not (20 <= age <= 79):
        return None

    ascvd_risk = 0

    # Risk parameters based on gender
    params = {
        "male": {
            "major2": 69,
            "major1": 50,
            "elevated": 46,
            "notOptimal": 36,
            "allOptimal": 5,
        },
        "female": {
            "major2": 50,
            "major1": 39,
            "elevated": 39,
            "notOptimal": 27,
            "allOptimal": 8,
        },
    }

       # Calculate major risk factors (fixed boolean checks)
    major = (1 if cholesterol_tot >= 240 else 0) + \
            (1 if bp_systolic >= 160 else 0) + \
            (1 if hypertension_treatment == "yes" else 0) + \
            (1 if smoking_status == "current" else 0) + \
            (1 if diabetes_status == "yes" else 0)

    # Rest of the calculations remain the same
    elevated = (
        (1 if (200 <= cholesterol_tot < 240) else 0) +
        (1 if (140 <= bp_systolic < 160 and hypertension_treatment == "no") else 0)
    )
    elevated = 1 if elevated >= 1 and major == 0 else 0

    all_optimal = (
        (1 if cholesterol_tot < 180 else 0) +
        (1 if bp_systolic < 120 and hypertension_treatment == "no" else 0)
    )
    all_optimal = 1 if all_optimal == 2 and major == 0 else 0

    not_optimal = (
        (1 if (180 <= cholesterol_tot < 200) else 0) +
        (1 if (120 <= bp_systolic < 140 and hypertension_treatment == "no") else 0)
    )
    not_optimal = 1 if not_optimal >= 1 and elevated == 0 and major == 0 else 0

    # Determine risk based on conditions
    if major > 1:
        ascvd_risk = params[gender]["major2"]
    elif major == 1:
        ascvd_risk = params[gender]["major1"]
    elif elevated == 1:
        ascvd_risk = params[gender]["elevated"]
    elif not_optimal == 1:
        ascvd_risk = params[gender]["notOptimal"]
    elif all_optimal == 1:
        ascvd_risk = params[gender]["allOptimal"]

    return ascvd_risk


def compute_SA_lifetime_risk(metrics):
    """
    Compute the lifetime ASCVD risk score for South Asian individuals.
    Applies a 1.5x multiplier to account for increased risk in South Asian populations.
    """
    age = metrics['age']
    gender = metrics['gender']
    cholesterol_tot = metrics['cholesterol_tot']
    cholesterol_hdl = metrics['cholesterol_hdl']
    bp_systolic = metrics['bp_systolic']
    diabetes_status = metrics['diabetes_status']
    smoking_status = metrics['smoking_status']
    hypertension_treatment = metrics['hypertension_treatment']

    if not (20 <= age <= 79):
        return None

    # Risk parameters based on gender
    params = {
        "male": {
            "major2": 69,
            "major1": 50,
            "elevated": 46,
            "notOptimal": 36,
            "allOptimal": 5,
        },
        "female": {
            "major2": 50,
            "major1": 39,
            "elevated": 39,
            "notOptimal": 27,
            "allOptimal": 8,
        },
    }

    # Calculate major risk factors (fixed boolean checks)
    major = (1 if cholesterol_tot >= 240 else 0) + \
            (1 if bp_systolic >= 160 else 0) + \
            (1 if hypertension_treatment == "yes" else 0) + \
            (1 if smoking_status == "current" else 0) + \
            (1 if diabetes_status == "yes" else 0)

    # Rest of the calculations remain the same
    elevated = (
        (1 if (200 <= cholesterol_tot < 240) else 0) +
        (1 if (140 <= bp_systolic < 160 and hypertension_treatment == "no") else 0)
    )
    elevated = 1 if elevated >= 1 and major == 0 else 0

    all_optimal = (
        (1 if cholesterol_tot < 180 else 0) +
        (1 if bp_systolic < 120 and hypertension_treatment == "no" else 0)
    )
    all_optimal = 1 if all_optimal == 2 and major == 0 else 0

    not_optimal = (
        (1 if (180 <= cholesterol_tot < 200) else 0) +
        (1 if (120 <= bp_systolic < 140 and hypertension_treatment == "no") else 0)
    )
    not_optimal = 1 if not_optimal >= 1 and elevated == 0 and major == 0 else 0

    # Determine risk based on conditions
    if major > 1:
        ascvd_risk = params[gender]["major2"]
    elif major == 1:
        ascvd_risk = params[gender]["major1"]
    elif elevated == 1:
        ascvd_risk = params[gender]["elevated"]
    elif not_optimal == 1:
        ascvd_risk = params[gender]["notOptimal"]
    elif all_optimal == 1:
        ascvd_risk = params[gender]["allOptimal"]

    # Apply South Asian adjustment factor
    return ascvd_risk * 1.5


def compute_lowest_ten_year(metrics):
    """
    Compute the lowest possible 10-year ASCVD risk under optimal conditions.
    """
    optimal_metrics = metrics.copy()
    optimal_metrics.update({
        "bp_systolic": 90,
        "cholesterol_tot": 130,
        "cholesterol_hdl": 100,
        "diabetes_status": "no",
        "smoking_status": "never",
        "hypertension_treatment": "no",
    })
    return compute_ten_year_score(optimal_metrics)


def compute_lowest_lifetime(metrics):
    """
    Compute the lowest possible lifetime ASCVD risk under optimal conditions.
    """
    optimal_metrics = metrics.copy()
    optimal_metrics.update({
        "bp_systolic": 90,
        "cholesterol_tot": 130,
        "cholesterol_hdl": 100,
        "diabetes_status": "no",
        "smoking_status": "never",
        "hypertension_treatment": "no",
    })
    return compute_lifetime_risk(optimal_metrics)


def compute_potential_risk(reductions, score_type, metrics):
    """
    Compute the potential risk reduction based on factors such as statin use, reduced systolic BP, aspirin, or quitting smoking.

    Parameters:
        reductions (list): List of factors influencing risk reduction (e.g., 'statin', 'sysBP', 'aspirin', 'smoker').
        score_type (str): Either 'ten' for 10-year risk or 'lifetime' for lifetime risk.
        age, gender, total_cholesterol, systolic_blood_pressure, hdl, diabetic, smoker, hypertensive: Inputs for the risk calculator.

    Returns:
        float: The reduced risk score.
    """
    if score_type == 'ten':
        computed_score = compute_ten_year_score(metrics)
        lowest_score = compute_lowest_ten_year(metrics)
    else:
        computed_score = compute_lifetime_risk(metrics)
        lowest_score = compute_lowest_lifetime(metrics)

    reduced_total_score = 0
    for reduction in reductions:
        if reduction == 'statin':
            reduced_total_score += (computed_score * 0.25)
        elif reduction == 'hypertension_treatment':
            sys_bp_reduction = computed_score - (computed_score * (0.7 ** ((metrics["bp_systolic"] - 140) / 10)))
            reduced_total_score += sys_bp_reduction
        elif reduction == 'aspirin':
            reduced_total_score += (computed_score * 0.1)
        elif reduction == 'quit_smoking':
            reduced_total_score += (computed_score * 0.15)

    final_score = max(computed_score - reduced_total_score, lowest_score)
    return round(final_score, 1)

def create_risk_slider(normal_score, sa_score, reductions, metrics):
    """
    Create a dynamic slider showing normal risk score and South Asian adjustment with animated segments
    """
    max_value = sa_score + 5  # Add padding for visualization
    
    # Calculate reduced scores if treatments are selected
    if reductions:
        reduced_normal = compute_potential_risk(reductions, 'ten' if metrics['age'] > 40 else 'lifetime', metrics)
        reduced_sa = reduced_normal * 1.5  # Apply South Asian adjustment
    else:
        reduced_normal = normal_score
        reduced_sa = sa_score
    
    # Helper function to get category color
    def get_category_color(score, is_sa=False):
        if metrics['age'] < 40:  # Lifetime risk colors
            if is_sa:
                return "#cc0000"  # Bright red for SA adjustment
            else:
                return "#0066cc"  # Bright blue for normal risk
        else:  # 10-year risk colors
            if score < 5:
                return "#28a745"  # Green
            elif score <= 7.4:
                return "#ffc107"  # Yellow
            elif score <= 19.9:
                return "#fd7e14"  # Orange
            else:
                return "#dc3545"  # Red

    # Create legend with dynamic colors based on risk scores
    legend_html = f"""
        <div style="display: flex; align-items: center; gap: 20px; margin-bottom: 10px;">
            <div style="display: flex; align-items: center;">
                <div style="width: 20px; height: 20px; background-color: {get_category_color(reduced_normal, False)}; margin-right: 5px;"></div>
                <span>Normal Risk</span>
            </div>
            <div style="display: flex; align-items: center;">
                <div style="width: 20px; height: 20px; background-color: {get_category_color(reduced_sa, True)}; margin-right: 5px;"></div>
                <span>South Asian Adjustment</span>
            </div>
        </div>
    """
    st.markdown(legend_html, unsafe_allow_html=True)
    
    # Add CSS for tooltips
    st.markdown("""
        <style>
        .risk-segment {
            position: absolute;
            height: 100%;
            transition: opacity 0.3s;
        }
        .risk-segment:hover {
            opacity: 0.8;
        }
        .tooltip {
            visibility: hidden;
            position: absolute;
            background-color: black;
            color: white;
            padding: 5px;
            border-radius: 4px;
            font-size: 12px;
            z-index: 100;
            top: -30px;
            left: 50%;
            transform: translateX(-50%);
            white-space: nowrap;
        }
        .risk-segment:hover .tooltip {
            visibility: visible;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Create progress bar container
    progress_container = st.empty()
    
    # Create interval markings
    intervals = list(range(0, int(max_value) + 5, 5))
    interval_marks = "".join([
        f'<div style="position: absolute; left: {(val / max_value * 100)}%; '
        f'transform: translateX(-50%); text-align: center;">'
        f'<div style="height: 5px; width: 1px; background-color: #666; margin: 0 auto;"></div>'
        f'<div style="font-size: 10px; margin-top: 5px;">{val}%</div>'
        f'</div>' for val in intervals
    ])
    
    # Animate the normal risk score (first segment)
    for i in range(11):
        current_normal = reduced_normal * i/10
        progress_html = f"""
        <div style="position: relative; width: 100%; height: 50px; margin-bottom: 20px;">
            <div style="position: relative; width: 100%; height: 30px; background-color: #f0f0f0; border-radius: 5px;">
                <div class="risk-segment" style="width: {(current_normal) / max_value * 100}%; 
                     background-color: {get_category_color(reduced_normal, False)}; 
                     border-radius: 5px;">
                    <div class="tooltip">Normal Risk: {round(current_normal, 1)}%</div>
                </div>
            </div>
            <div style="position: absolute; width: 100%; top: 30px; height: 20px;">
                {interval_marks}
            </div>
        </div>
        """
        progress_container.markdown(progress_html, unsafe_allow_html=True)
        time.sleep(0.05)
    
    # Animate the South Asian adjustment (second segment)
    for i in range(11):
        additional = (reduced_sa - reduced_normal) * i/10
        current_sa = reduced_normal + additional
        progress_html = f"""
        <div style="position: relative; width: 100%; height: 50px; margin-bottom: 20px;">
            <div style="position: relative; width: 100%; height: 30px; background-color: #f0f0f0; border-radius: 5px;">
                <div class="risk-segment" style="width: {(reduced_normal) / max_value * 100}%; 
                     background-color: {get_category_color(reduced_normal, False)}; 
                     border-radius: 5px 0px 0px 5px;">
                    <div class="tooltip">Normal Risk: {round(reduced_normal, 1)}%</div>
                </div>
                <div style="position: absolute; 
                     left: {(reduced_normal) / max_value * 100}%;
                     height: 100%;
                     width: 2px;
                     background-color: #333;
                     z-index: 2;">
                </div>
                <div class="risk-segment" style="left: {(reduced_normal) / max_value * 100}%; 
                     width: {(additional) / max_value * 100}%; 
                     background-color: {get_category_color(reduced_sa, True)}; 
                     border-radius: 0px 5px 5px 0px;">
                    <div class="tooltip">SA Adjusted Risk: {round(current_sa, 1)}%</div>
                </div>
            </div>
            <div style="position: absolute; width: 100%; top: 30px; height: 20px;">
                {interval_marks}
            </div>
        </div>
        """
        progress_container.markdown(progress_html, unsafe_allow_html=True)
        time.sleep(0.05)

def get_risk_category(score):
    """Helper function to determine risk category"""
    if score < 5:
        return "Low-risk"
    elif score <= 7.4:
        return "Borderline risk"
    elif score <= 19.9:
        return "Intermediate risk"
    else:
        return "High risk"


# Input form
age = st.slider("Enter Age (20-79):", 20, 79, 50)
gender = st.selectbox("Gender:", ["male", "female"])
BMI = st.slider("Enter BMI", 10, 50, 22)
cholesterol_tot = st.slider("Total Cholesterol (mg/dL):", 130, 320, 200)
cholesterol_hdl = st.slider("HDL Cholesterol (mg/dL):", 20, 100, 50)
bp_systolic = st.slider("Systolic Blood Pressure (mm Hg):", 90, 200, 120)
diabetes_status = st.selectbox("Diabetes Status:", ["yes", "no"])
smoking_status = st.selectbox("Smoking Status:", ["never", "former", "current"])
hypertension_treatment = st.selectbox("On BP Meds:", ["yes", "no"])

# Compile metrics
metrics = {
    "age": age,
    "gender": gender,
    "cholesterol_tot": cholesterol_tot,
    "cholesterol_hdl": cholesterol_hdl,
    "bp_systolic": bp_systolic,
    "diabetes_status": diabetes_status,
    "smoking_status": smoking_status,
    "hypertension_treatment": hypertension_treatment
}


# Add separator and heading for visualization options
st.markdown("---")
st.write("**Please select which format you would like to see your risk score displayed:**")

# Create columns for the buttons
col1, col2, col3, col4 = st.columns(4)

# Button callbacks to update session state
def show_text_view():
    st.session_state.current_view = 'text'

def show_slider_view():
    st.session_state.current_view = 'slider'

def show_graph_view():
    st.session_state.current_view = 'graph'

def show_map_view():
    st.session_state.current_view = 'map'

with col1:
    st.button("Text Explanation", on_click=show_text_view)
with col2:
    st.button("Risk Slider", on_click=show_slider_view)
with col3:
    st.button("Risk Histogram", on_click=show_graph_view)
with col4:
    st.button("Risk Heatmap", on_click=show_map_view)

# Calculate scores if a view is selected
if st.session_state.current_view:
    if age < 40:
        risk_type = "Lifetime Risk"
        normal_score = compute_lifetime_risk(metrics)
        sa_score = compute_SA_lifetime_risk(metrics)
    else:
        risk_type = "10-Year Risk"
        normal_score = compute_ten_year_score(metrics)
        sa_score = compute_SA_ten_year_score(metrics)

    # Display container for results
    st.markdown("---")
    result_container = st.container()

    with result_container:
        
        recommendations = []
        
        # Check cholesterol
        if metrics['cholesterol_tot'] > 240:
            recommendations.append("🔸 You may want to consult your doctor for statin treatment given your high cholesterol levels.")
            
        # Check blood pressure
        if metrics['bp_systolic'] > 160:
            recommendations.append("🔸 You may want to consult your doctor for hypertension treatment given your high sysBP level.")
            
        # Check smoking status
        if metrics['smoking_status'] == "current":
            recommendations.append("🔸 CVD risk is greatly amplified by smoking, you may want to consult your doctor on a plan to quit smoking.")
            
        # Check diabetes status
        if metrics['diabetes_status'] == "yes":
            recommendations.append("🔸 You may want to consult your doctor for aspirin treatment given that you are diabetic.")
        
        # Add age-specific recommendations for patients under 40
        if metrics['age'] < 40:
            if sa_score > 50:
                recommendations.append("🔸 You have a high lifetime risk for CVD, you may want to consult your doctor and get a screening done.")
            elif 20 <= sa_score <= 50:
                recommendations.append("🔸 You have a moderate lifetime risk for CVD, you may want to consult your doctor and get a screening done.")
            else:  # sa_score < 20
                recommendations.append("🔸 You currently show low lifetime risk for CVD. You may want to consider consulting your doctor and getting screened due to ethnicity-compounded risk factors, or use this risk calculator at a later date.")
        
        # Display recommendations if any exist
        if recommendations:
            for rec in recommendations:
                st.write(rec)
        else:
            st.write("No specific recommendations at this time. Continue maintaining your current healthy lifestyle!")
        # Define and display reduction options in results section
        available_reductions = ["statin", "aspirin"]
        if metrics["bp_systolic"] > 140:
            available_reductions.append("hypertension_treatment")
        if metrics["smoking_status"] == "current":
            available_reductions.append("quit_smoking")

        reductions = st.multiselect(
            "Select interventions to see potential risk reduction:",
            available_reductions
        )

        # Calculate reduced scores if any reductions selected
        if reductions:
            reduced_normal = compute_potential_risk(reductions, 'ten' if age > 40 else 'lifetime', metrics)
            reduced_sa = reduced_normal * 1.5
        else:
            reduced_normal = normal_score
            reduced_sa = sa_score

        # South Asians
        np.random.seed(42)
        sa_risk_scores = np.random.normal(28, 8, 8300)  # Mean ~28, std dev 8
        sa_outcomes = np.random.binomial(1, 0.9, 8300) * sa_risk_scores + np.random.normal(5, 5, 8300)

        # Non-South Asians
        non_sa_risk_scores = np.random.normal(33, 10, 8300)  # Mean ~33, std dev 10
        non_sa_outcomes = np.random.binomial(1, 0.2, 8300) * non_sa_risk_scores + np.random.normal(5, 10, 8300)

        # data for South Asians
        sa_bp = np.random.randint(90, 151, 500)  # BP between 90 and 150
        sa_chol = np.random.randint(135, 301, 500)  # Cholesterol between 135 and 300
        sa_risk = (0.003 * sa_bp * sa_chol / 100) + np.clip((sa_bp * 0.4 + sa_chol * 0.6) / 10, 0, 50)

        # data for Non-South Asians
        non_sa_bp = np.random.randint(90, 151, 500)  # BP between 90 and 150
        non_sa_chol = np.random.randint(135, 301, 500)  # Cholesterol between 135 and 300
        non_sa_risk = (0.002 * non_sa_bp * non_sa_chol / 100) + np.clip((non_sa_bp * 0.3 + non_sa_chol * 0.7) / 10, 0, 50)
        
        if st.session_state.current_view == 'text':
            st.subheader("Risk Assessment Explanation")
            st.write(f"Your {risk_type} assessment shows:")
            
            # Base risk categorization
            base_category = get_risk_category(normal_score)
            st.write(f"- Base Risk Score: {normal_score}% ({base_category})")
            
            # SA adjusted risk categorization
            sa_category = get_risk_category(sa_score)
            st.write(f"- South Asian Adjusted Risk Score: {sa_score}% ({sa_category})")
            
            if reductions:
                # Reduced risk categorization
                reduced_base_category = get_risk_category(reduced_normal)
                reduced_sa_category = get_risk_category(reduced_sa)
                
                st.write(f"With your selected interventions:")
                st.write(f"- Reduced Base Risk: {reduced_normal}% ({reduced_base_category})")
                st.write(f"- Reduced South Asian Adjusted Risk: {reduced_sa}% ({reduced_sa_category})")
                
                reduction_text = ", ".join(reductions)
                st.write(f"Based on: {reduction_text}")

        elif st.session_state.current_view == 'slider':
            st.subheader("Risk Score Visualization")
            if reductions:
                create_risk_slider(reduced_normal, reduced_sa, [], metrics)
                if age >= 40:  # Only show risk category for 10-year risk
                    st.write(f"Current risk category: **{get_risk_category(reduced_sa)}**")
            else:
                create_risk_slider(normal_score, sa_score, [], metrics)
                if age >= 40:  # Only show risk category for 10-year risk
                    st.write(f"Current risk category: **{get_risk_category(sa_score)}**")
            
            # Add risk scale reference only for 10-year risk
            if age >= 40:
                st.markdown("---")
                st.write("**Risk Categories:**")
                cols = st.columns(4)
                with cols[0]:
                    st.markdown("🟢 **Low**\n<5%", unsafe_allow_html=True)
                with cols[1]:
                    st.markdown("🟡 **Borderline**\n5-7.4%", unsafe_allow_html=True)
                with cols[2]:
                    st.markdown("🟠 **Intermediate**\n7.5-19.9%", unsafe_allow_html=True)
                with cols[3]:
                    st.markdown("🔴 **High**\n≥20%", unsafe_allow_html=True)

        elif st.session_state.current_view == 'graph':
            import streamlit as st
            import numpy as np
            import matplotlib.pyplot as plt

            you_risk_score = normal_score * 3
            you_outcome = 30

            # Function to plot histogram
            def plot_risk_histograms():
                # Set transparent background and font styles
                plt.style.use("default")
                plt.rcParams.update({
                    "axes.facecolor": "none",  # Transparent background for the plot
                    "figure.facecolor": "none",  # Transparent background for the figure
                    "text.color": "#262730",  # Text color to match Streamlit's default
                    "axes.labelcolor": "#262730",
                    "xtick.color": "#262730",
                    "ytick.color": "#262730",
                    "font.size": 12,
                    "font.family": "sans-serif",  # Matches Streamlit's default font
                })

                # Create the plot
                plt.figure(figsize=(10, 6))
                bins = np.linspace(10, 50, 30)
                
                # South Asians risk scores
                plt.hist(sa_risk_scores, bins=bins, alpha=0.6, color='blue', label="South Asians")
                plt.hist(non_sa_risk_scores, bins=bins, alpha=0.6, color='orange', label="Non-South Asians")
                
                bucket_index = np.digitize(you_risk_score, bins) - 1
                if bucket_index < len(bins) - 1:
                    plt.axvspan(bins[bucket_index], bins[bucket_index + 1], color='red', alpha=0.2, label="Your Bucket")

                # Labels and legend
                plt.title("Risk Scores vs Frequency of Negative Outcomes", fontsize=14)
                plt.xlabel("Risk Score", fontsize=12)
                plt.ylabel("Frequency", fontsize=12)
                plt.axvline(30, color='gray', linestyle='--', label="Mean Risk Score")
                plt.legend()
                plt.tight_layout()
                
                # Return the figure
                return plt

            # Streamlit App
            st.subheader("Risk Score Histogram")

            # Generate and display the plot
            fig = plot_risk_histograms()
            st.pyplot(fig, transparent=True)

        elif st.session_state.current_view == 'map':
            st.subheader("Risk Score Heatmap")
            import numpy as np
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Scale the risk scores to a range of 0 to 50
            sa_risk = (sa_risk - sa_risk.min()) / (sa_risk.max() - sa_risk.min()) * 500
            non_sa_risk = (non_sa_risk - non_sa_risk.min()) / (non_sa_risk.max() - non_sa_risk.min()) * 500

            # Define heatmap bins
            bp_bins = np.linspace(90, 150, 30)
            chol_bins = np.linspace(135, 250, 30)

            # Create heatmap grids
            sa_heatmap, _, _ = np.histogram2d(sa_bp, sa_chol, bins=[bp_bins, chol_bins], weights=sa_risk)
            non_sa_heatmap, _, _ = np.histogram2d(non_sa_bp, non_sa_chol, bins=[bp_bins, chol_bins], weights=non_sa_risk)

            # Normalize the heatmaps
            sa_heatmap_normalized = sa_heatmap / np.max(sa_heatmap)
            non_sa_heatmap_normalized = non_sa_heatmap / np.max(non_sa_heatmap)

            # Add gradual increase for South Asians
            for i in range(len(bp_bins) - 1):
                for j in range(len(chol_bins) - 1):
                    if bp_bins[i] > 115 and chol_bins[j] > 130:
                        sa_heatmap_normalized[i, j] += 0.25 * (bp_bins[i] - 90) * (chol_bins[j] - 120) / 1000

            for i in range(len(bp_bins) - 1):
                for j in range(len(chol_bins) - 1):
                    if (bp_bins[i] > 85 and bp_bins[i] < 115) or (chol_bins[j] > 100 and chol_bins[j] < 130):
                        sa_heatmap_normalized[i, j] += 0.15 * (bp_bins[i] - 75) * (chol_bins[j] - 80) / 1000


            for i in range(len(bp_bins) - 1):
                for j in range(len(chol_bins) - 1):
                    if bp_bins[i] > 137 and chol_bins[j] > 200:
                        non_sa_heatmap_normalized[i, j] += 0.25 * (bp_bins[i] - 115) * (chol_bins[j] - 160) / 1000

            for i in range(len(bp_bins) - 1):
                for j in range(len(chol_bins) - 1):
                    if (bp_bins[i] > 115 and bp_bins[i] < 137) or (chol_bins[j] > 180 and chol_bins[j] < 200):
                        non_sa_heatmap_normalized[i, j] += 0.1 * (bp_bins[i] - 105) * (chol_bins[j] - 170) / 1000


            # Amplify normalized values for both groups
            for i in range(len(bp_bins) - 1):
                for j in range(len(chol_bins) - 1):
                    sa_heatmap_normalized[i, j] = 20 * sa_heatmap_normalized[i, j]
                    non_sa_heatmap_normalized[i, j] = 34 * non_sa_heatmap_normalized[i, j]

            # Define user input point
            user_bp = metrics['bp_systolic']
            user_chol = metrics['cholesterol_tot']

            # Adjust the layout
            plt.tight_layout()
            plt.show()

            # Plot function
            def plot_heatmaps():
                # Set transparent background and style
                plt.style.use("default")
                plt.rcParams.update({
                    "axes.facecolor": "none",  # Transparent plot background
                    "figure.facecolor": "none",  # Transparent figure background
                    "text.color": "#262730",  # Match Streamlit's text color
                    "axes.labelcolor": "#262730",
                    "xtick.color": "#262730",
                    "ytick.color": "#262730",
                    "font.size": 12,
                    "font.family": "sans-serif",
                })

                # Create the figure and axes
                fig, axes = plt.subplots(1, 2, figsize=(28, 10), sharex=True, sharey=True)

                # South Asian heatmap with gradual transition
                sns.heatmap(sa_heatmap_normalized.T, ax=axes[0], cmap="YlGnBu", cbar=True,
                            xticklabels=np.round(bp_bins[1:], 1), yticklabels=np.round(chol_bins[1:], 1))
                axes[0].set_title("South Asian Risk Distribution", fontsize=24)
                axes[0].set_xlabel("Systolic Blood Pressure (mm Hg)")
                axes[0].set_ylabel("Total Cholesterol (mg/dL)")
                axes[0].add_patch(plt.Rectangle((np.digitize(user_bp, bp_bins) - 1, 
                                                  np.digitize(user_chol, chol_bins) - 1), 
                                                 1, 1, fill=False, edgecolor='red', linewidth=2, label="User"))

                # Non-South Asian heatmap with gradual transition
                sns.heatmap(non_sa_heatmap_normalized.T, ax=axes[1], cmap="YlGnBu", cbar=True,
                            xticklabels=np.round(bp_bins[1:], 1), yticklabels=np.round(chol_bins[1:], 1))
                axes[1].set_title("Non-South Asian Risk Distribution", fontsize=24)
                axes[1].set_xlabel("Systolic Blood Pressure (mm Hg)")
                axes[1].add_patch(plt.Rectangle((np.digitize(user_bp, bp_bins) - 1, 
                                                  np.digitize(user_chol, chol_bins) - 1), 
                                                 1, 1, fill=False, edgecolor='red', linewidth=2, label="User"))


                # Adjust layout and return
                plt.tight_layout()
                return fig

            # Generate and display the plot in Streamlit
            fig = plot_heatmaps()
            st.pyplot(fig, transparent=True)

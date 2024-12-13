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
    """
    age = metrics['age']
    gender = metrics['gender']
    cholesterol_tot = metrics['cholesterol_tot']
    bp_systolic = metrics['bp_systolic']
    hypertension_treatment = metrics['hypertension_treatment']
    smoking_status = metrics['smoking_status']
    diabetes_status = metrics['diabetes_status']

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

    # Count major risk factors
    major_count = 0
    if cholesterol_tot >= 240:
        major_count += 1
    if bp_systolic >= 160:
        major_count += 1
    if hypertension_treatment == "yes":
        major_count += 1
    if smoking_status == "current":
        major_count += 1
    if diabetes_status == "yes":
        major_count += 1

    # Check for elevated risk factors (only if no major risk factors)
    elevated = False
    if major_count == 0:
        if 200 < cholesterol_tot < 240:
            elevated = True
        if 140 <= bp_systolic < 160 and hypertension_treatment == "no":
            elevated = True

    # Check for optimal factors
    all_optimal = (cholesterol_tot < 180 and 
                  bp_systolic < 120 and 
                  hypertension_treatment == "no" and
                  smoking_status == "never" and
                  diabetes_status == "no")

    # Check for not optimal factors
    not_optimal = False
    if not elevated and major_count == 0 and not all_optimal:
        if 180 <= cholesterol_tot < 200:
            not_optimal = True
        if 120 <= bp_systolic < 140 and hypertension_treatment == "no":
            not_optimal = True

    # Determine risk based on conditions
    if major_count > 1:
        return params[gender]["major2"]
    elif major_count == 1:
        return params[gender]["major1"]
    elif elevated:
        return params[gender]["elevated"]
    elif not_optimal:
        return params[gender]["notOptimal"]
    elif all_optimal:
        return params[gender]["allOptimal"]
    else:
        return params[gender]["notOptimal"]  # Default to not optimal if no other conditions met


def compute_SA_lifetime_risk(metrics):
    """
    Compute the lifetime ASCVD risk score for South Asian individuals.
    Applies a 1.5x multiplier to account for increased risk in South Asian populations.
    """
    base_risk = compute_lifetime_risk(metrics)
    if base_risk is not None:
        return base_risk * 1.5
    return None


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
col1, col2, col3 = st.columns(3)

# Button callbacks to update session state
def show_text_view():
    st.session_state.current_view = 'text'

def show_slider_view():
    st.session_state.current_view = 'slider'

def show_graph_view():
    st.session_state.current_view = 'graph'

with col1:
    st.button("Text Explanation", on_click=show_text_view)
with col2:
    st.button("Risk Slider", on_click=show_slider_view)
with col3:
    st.button("Risk Graph", on_click=show_graph_view)

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
        # Add explanatory notes based on age
        if age < 40:
            st.info("""
                📋 **Note on Lifetime Risk Assessment:**
                - You are under 40, so only lifetime risk calculation is applicable
                - Lifetime risk is the probability of developing CVD over the next 30 years. 
                - Risk scores will appear higher than 10-year risk due to the longer time horizon and greater uncertainty
                - This calculation helps identify long-term cardiovascular risk for early intervention
            """)
        else:
            st.info("""
                📋 **Note on 10-Year Risk Assessment:**
                - You are 40 or older, so a 10-year risk calculation is used. This is the probability of developing CVD over the next 10 years. 
                - This is a more immediate assessment of cardiovascular risk
            """)
            
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
            st.subheader("Risk Score Distribution")
            
            # Generate range of values for simulation
            bp_range = range(90, 201, 10)
            chol_range = range(130, 321, 10)
            age_range = range(max(20, age-10), min(79, age+11), 1)
            
            # Create simulated scores based on age variation
            normal_scores = []
            sa_adjusted_scores = []
            
            for test_age in age_range:
                test_metrics = metrics.copy()
                test_metrics['age'] = test_age
                
                if age < 40:
                    normal_score = compute_lifetime_risk(test_metrics)
                    sa_score = compute_SA_lifetime_risk(test_metrics)
                else:
                    normal_score = compute_ten_year_score(test_metrics)
                    sa_score = compute_SA_ten_year_score(test_metrics)
                    
                if normal_score is not None and sa_score is not None:
                    normal_scores.append(normal_score)
                    sa_adjusted_scores.append(sa_score)
            
            # Create and display the histogram
            fig = plot_overlayed_histogram(normal_scores, sa_adjusted_scores, "Age-based Risk Distribution")
            st.plotly_chart(fig)
            
            if reductions:
                if age >= 40:  # Only show risk category for 10-year risk
                    st.write(f"Current risk category with interventions: **{get_risk_category(reduced_sa)}**")
            else:
                if age >= 40:  # Only show risk category for 10-year risk
                    st.write(f"Current risk category: **{get_risk_category(sa_score)}**")
            
            # Add risk scale reference only for 10-year risk
            if age >= 40:
                st.markdown("---")
                st.write("**ASCVD Risk Categories:**")
                cols = st.columns(4)
                with cols[0]:
                    st.markdown("🟢 **Low**\n<5%", unsafe_allow_html=True)
                with cols[1]:
                    st.markdown("🟡 **Borderline**\n5-7.4%", unsafe_allow_html=True)
                with cols[2]:
                    st.markdown("🟠 **Intermediate**\n7.5-19.9%", unsafe_allow_html=True)
                with cols[3]:
                    st.markdown("🔴 **High**\n≥20%", unsafe_allow_html=True)




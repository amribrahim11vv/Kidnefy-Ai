import streamlit as st
import sys
from pathlib import Path
import os
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing import calculate_egfr
from src.staging import GFRCalculator, RiskAssessor
from src.models.staging_model import StagingModel

# Set page layout
st.set_page_config(
    page_title="تطبيق التنبؤ بأمراض الكلى (Kidney Disease Prediction)", 
    page_icon="", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom RTL style
st.markdown("""
<style>
    body {
        direction: rtl;
        text-align: right;
        font-family: 'Cairo', sans-serif, 'Segoe UI';
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #007bff;
    }
    .metric-label {
        font-size: 16px;
        color: #6c757d;
        margin-bottom: 5px;
    }
    div[data-testid="stSidebar"] {
        direction: rtl;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.title(" نظام التنبؤ بأمراض الكلى المتقدم")
st.markdown("### Advanced Kidney Disease Prediction System")
st.markdown("هذا التطبيق يوضح آلية عمل نماذج الذكاء الاصطناعي والأنظمة الخبيرة الخاصة بمشروعك للتنبؤ وتقييم مراحل الكلى.")

# Load models safely
@st.cache_resource
def load_models():
    gfr_calc = GFRCalculator()
    risk_assess = RiskAssessor()
    try:
        staging_mod = StagingModel()
    except Exception as e:
        staging_mod = None
        st.sidebar.warning(f"تم إيقاف نموذج StagingModel مؤقتاً: {e}")
    return gfr_calc, risk_assess, staging_mod

gfr_calculator, risk_assessor, staging_model = load_models()

# Sidebar for Patient Information
with st.sidebar:
    st.header(" بيانات المريض (Patient Profile)")
    name = st.text_input("الاسم (Name)", "مريض تجريبي")
    age = st.number_input("العمر (Age)", min_value=1, max_value=120, value=55)
    
    sex_options = ["ذكر (Male)", "أنثى (Female)"]
    sex_input = st.radio("الجنس (Sex)", sex_options)
    is_female = "Female" in sex_input
    
    st.markdown("---")
    st.info(" **ملاحظة:** يتم حساب eGFR باستخدام معادلة CKD-EPI 2021 التي تم دمجها في نظامك.")

# Main Form for Medical Tests
st.header(" القراءات المخبرية (Lab Results)")

col1, col2, col3 = st.columns(3)

with col1:
    creatinine = st.number_input("كرياتينين الدم (Creatinine) mg/dL", min_value=0.1, max_value=20.0, value=1.2, step=0.1)
    blood_glucose = st.number_input("سكر الدم (Blood Glucose) mg/dL", min_value=50.0, max_value=1000.0, value=110.0, step=10.0)

with col2:
    acr = st.number_input("معدل الألبومين/الكرياتينين (ACR) mg/g", min_value=0.0, max_value=5000.0, value=15.0, step=5.0)
    blood_urea = st.number_input("يوريا الدم (BUN) mg/dL", min_value=1.0, max_value=300.0, value=20.0, step=5.0)

with col3:
    hemoglobin = st.number_input("الهيموجلوبين (Hemoglobin) g/dL", min_value=1.0, max_value=25.0, value=14.0, step=0.5)
    blood_pressure = st.number_input("الضغط الانقباضي (Systolic BP)", min_value=50, max_value=250, value=120, step=5)

submit_btn = st.button("توقع النتائج وعرض المرحلة (Predict & Stage)", type="primary", use_container_width=True)

st.markdown("---")

if submit_btn:
    with st.spinner("جاري التحليل باستخدام النماذج الذكية..."):
        
        # 1. Calculate eGFR
        egfr = gfr_calculator.calculate_egfr_ckdepi(creatinine, age, is_female)
        
        # 2. Complete Staging via Guidelines
        staging = gfr_calculator.calculate_stage(
            creatinine=creatinine,
            acr=acr,
            age=age,
            is_female=is_female
        )
        
        # 3. AI Staging Prediction
        ai_stage_result = None
        if staging_model:
            # The model expects 12 features: serum_creatinine, gfr, bun, serum_calcium, ana, c3_c4,
            # hematuria, oxalate_levels, urine_ph, blood_pressure, water_intake, months
            # We map user inputs and provide medical defaults for the rest to avoid 0-filling
            stage_input = {
                "serum_creatinine": creatinine,
                "gfr": egfr, # Extremely important! Without this it defaults to 0 (Stage 5)
                "bun": blood_urea,
                "serum_calcium": 9.5, # Normal Calcium
                "ana": 0.5,           # Normal ANA
                "c3_c4": 15.0,        # Normal C3/C4
                "hematuria": 0.0,     # No hematuria
                "oxalate_levels": 20.0, # Normal Oxalate
                "urine_ph": 6.0,      # Normal pH
                "blood_pressure": blood_pressure,
                "water_intake": 2.0,  # Average 2L/day
                "months": 1.0         # Default time tracking
            }
            try:
                ai_stage_result = staging_model.predict_stage(stage_input)
            except Exception as e:
                pass
        
        # DISPLAY RESULTS
        st.header(" النتائج التحليلية (Analysis Results)")
        
        # Metrics row
        m1, m2, m3, m4 = st.columns(4)
        
        with m1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">eGFR (وظائف الكلى)</div><div class="metric-value">{egfr:.1f}</div><div>mL/min/1.73m²</div></div>', unsafe_allow_html=True)
            
        with m2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">المرحلة السريرية (Stage)</div><div class="metric-value">{staging.gfr_stage.value}</div><div>حسب KDIGO</div></div>', unsafe_allow_html=True)
            
        with m3:
            alb_cat = staging.albuminuria_category.value if staging.albuminuria_category else "غير متوفر"
            st.markdown(f'<div class="metric-card"><div class="metric-label">مستوى الزلال (Albuminuria)</div><div class="metric-value">{alb_cat}</div><div>مستوى البروتين</div></div>', unsafe_allow_html=True)
            
        with m4:
            # Color coding for risk
            risk_color = "green" if "Low" in staging.risk_level.value else ("orange" if "Moderate" in staging.risk_level.value else "red")
            st.markdown(f'<div class="metric-card"><div class="metric-label">مستوى الخطر (Risk Level)</div><div class="metric-value" style="color: {risk_color}">{staging.risk_level.value}</div><div>احتمالية التطور</div></div>', unsafe_allow_html=True)

        # AI Prediction Details
        st.subheader(" تحليل الذكاء الاصطناعي (AI Analysis Model)")
        if ai_stage_result:
            pred_stage = ai_stage_result.get("predicted_stage", "غير معروف")
            confidence = ai_stage_result.get("confidence", 0.0)
            
            # Message mapping
            st.success(f"**رؤية نموذج الذكاء الاصطناعي (AI Staging Model):** يتوقع النموذج أن المريض في مرحلة **{pred_stage}** بدقة تقديرية تبلغ **{confidence:.1%}**.")
            
            with st.expander("تفاصيل احتمالات الذكاء الاصطناعي (Probabilities Breakdown)", expanded=False):
                probs = ai_stage_result.get("probabilities", {})
                
                # Plotting bar chart of probabilities
                if probs:
                    fig, ax = plt.subplots(figsize=(8, 3))
                    stages = list(probs.keys())
                    values = [v * 100 for v in probs.values()]
                    ax.bar(stages, values, color='#007bff')
                    ax.set_ylabel('Probability (%)')
                    ax.set_title('AI Stage Probabilities')
                    ax.set_ylim(0, 100)
                    for i, v in enumerate(values):
                        ax.text(i, v + 2, f'{v:.1f}%', ha='center')
                    st.pyplot(fig)
        else:
            st.info("نموذج الذكاء الاصطناعي لا يتم تشغيله حالياً أو بعض المدخلات ناقصة.")

        st.subheader(" التقرير الطبي والتوصيات")
        st.write(f"**التفسير الطبي:** {staging.description}")
        
        if staging.recommendations:
            st.write("**التوصيات:**")
            for rec in staging.recommendations:
                st.markdown(f"- ✅ {rec}")

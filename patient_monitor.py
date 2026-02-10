import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

try:
    from dotenv import load_dotenv
    from langchain_experimental.agents import create_csv_agent
except ImportError:
    st.error("❌ Missing packages. Run: pip install langchain-experimental python-dotenv tabulate plotly openai langchain-openai")
    st.stop()

# --- FIX: Import with fallback to handle proxies error ---
try:
    from langchain_openai import ChatOpenAI
    # Test instantiation to catch proxies error early
    _test_works = True
except Exception:
    _test_works = False

# --- LOAD ENVIRONMENT ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="AI ICU Patient Monitor | CDSS",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PROFESSIONAL MEDICAL DASHBOARD CSS
# ============================================================
st.markdown("""
<style>
    /* ---- GLOBAL ---- */
    .stApp {
        background-color: #f0f2f6;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }

    /* ---- SIDEBAR — LIGHT CLEAN THEME ---- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #e8edf2 100%) !important;
        border-right: 1px solid #d1d9e6;
    }
    section[data-testid="stSidebar"] * {
        color: #2d3436 !important;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #0c2461 !important;
        font-weight: 700 !important;
    }
    section[data-testid="stSidebar"] .stMarkdown p {
        color: #636e72 !important;
    }
    section[data-testid="stSidebar"] .stRadio label span {
        color: #2d3436 !important;
        font-weight: 500 !important;
    }
    section[data-testid="stSidebar"] .stSelectbox label {
        color: #2d3436 !important;
        font-weight: 600 !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #d1d9e6 !important;
    }

    /* Sidebar telemetry box */
    .sidebar-telemetry {
        background: #f0f4f8;
        padding: 10px 12px;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        font-size: 0.72em;
        line-height: 1.8;
        margin-top: 8px;
        border: 1px solid #d1d9e6;
    }
    .sidebar-telemetry .tele-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .sidebar-telemetry .tele-label {
        color: #636e72 !important;
        font-size: 0.85em;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        font-weight: 600;
    }
    .sidebar-telemetry .tele-value {
        color: #1e3799 !important;
        font-weight: 700;
        font-size: 0.95em;
    }

    /* ---- TOP HEADER BAR ---- */
    .dashboard-header {
        background: linear-gradient(90deg, #0c2461, #1e3799, #4a69bd);
        padding: 18px 30px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    .dashboard-header .title-section h1 {
        margin: 0; font-size: 1.6em; font-weight: 700; color: #ffffff;
        letter-spacing: 0.5px;
    }
    .dashboard-header .title-section p {
        margin: 2px 0 0 0; font-size: 0.85em; color: #a4c4f4;
    }
    .header-badge {
        background: #00b894; color: white; padding: 6px 16px;
        border-radius: 20px; font-size: 0.8em; font-weight: 600;
        letter-spacing: 0.5px;
    }

    /* ---- PATIENT INFO BANNER ---- */
    .patient-banner {
        background: #ffffff;
        border-radius: 12px;
        padding: 20px 28px;
        margin-bottom: 20px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        border-left: 5px solid #1e3799;
        display: flex;
        flex-wrap: wrap;
        gap: 30px;
        align-items: center;
    }
    .patient-banner .patient-avatar {
        width: 60px; height: 60px; border-radius: 50%;
        background: linear-gradient(135deg, #1e3799, #4a69bd);
        display: flex; align-items: center; justify-content: center;
        font-size: 1.6em; color: white; font-weight: 700;
        flex-shrink: 0;
    }
    .patient-banner .info-group {
        display: flex; flex-direction: column; gap: 2px;
    }
    .patient-banner .info-label {
        font-size: 0.7em; color: #636e72; text-transform: uppercase;
        font-weight: 600; letter-spacing: 0.8px;
    }
    .patient-banner .info-value {
        font-size: 1.0em; color: #2d3436; font-weight: 600;
    }

    /* ---- VITAL SIGN CARDS (COMPACT) ---- */
    .vital-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 10px 8px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border-top: 3px solid #dfe6e9;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        min-height: 120px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .vital-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 14px rgba(0,0,0,0.09);
    }
    .vital-card .vital-icon { font-size: 1.1em; margin-bottom: 2px; }
    .vital-card .vital-label {
        font-size: 0.6em; color: #636e72; text-transform: uppercase;
        font-weight: 600; letter-spacing: 0.5px; margin-bottom: 2px;
    }
    .vital-card .vital-value {
        font-size: 1.3em; font-weight: 800; color: #2d3436; line-height: 1.15;
        word-break: break-word;
        overflow-wrap: break-word;
        max-width: 100%;
    }
    .vital-card .vital-unit { font-size: 0.55em; color: #636e72; font-weight: 500; }
    .vital-card .vital-range { font-size: 0.55em; color: #b2bec3; margin-top: 3px; }

    .vital-card.normal  { border-top-color: #00b894; }
    .vital-card.warning { border-top-color: #fdcb6e; }
    .vital-card.critical { border-top-color: #d63031; }
    .vital-card.critical .vital-value { color: #d63031; }
    .vital-card.warning .vital-value  { color: #e17055; }
    .vital-card.normal .vital-value   { color: #00b894; }

    /* ---- STATUS DOT ---- */
    .status-dot {
        display: inline-block; width: 8px; height: 8px;
        border-radius: 50%; margin-right: 4px;
    }
    .status-dot.green  { background: #00b894; }
    .status-dot.yellow { background: #fdcb6e; }
    .status-dot.red    { background: #d63031; animation: blink 1s infinite; }

    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }

    /* ---- CRITICAL ALERT BANNER ---- */
    .critical-alert-banner {
        background: linear-gradient(90deg, #d63031, #e74c3c);
        color: white;
        padding: 16px 24px;
        border-radius: 12px;
        text-align: center;
        font-size: 1.1em;
        font-weight: 700;
        margin-bottom: 20px;
        animation: pulse-alert 1.5s ease-in-out infinite;
        box-shadow: 0 0 30px rgba(214,48,49,0.4);
        letter-spacing: 0.5px;
    }
    @keyframes pulse-alert {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.85; }
    }

    /* ---- SECTION HEADINGS ---- */
    .section-header {
        font-size: 1.1em; font-weight: 700; color: #2d3436;
        margin: 24px 0 12px 0; padding-bottom: 8px;
        border-bottom: 2px solid #dfe6e9;
        display: flex; align-items: center; gap: 8px;
    }

    /* ---- AI RESPONSE BOX ---- */
    .ai-response-box {
        background: linear-gradient(135deg, #f8f9fa, #ffffff);
        border-left: 5px solid #1e3799;
        padding: 20px 24px;
        border-radius: 10px;
        margin: 16px 0;
        box-shadow: 0 2px 12px rgba(0,0,0,0.05);
        font-size: 0.95em;
        line-height: 1.7;
        color: #2d3436;
    }

    /* ---- BUTTONS ---- */
    .stButton > button {
        background: linear-gradient(90deg, #1e3799, #4a69bd);
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 8px 24px;
        font-weight: 600;
        letter-spacing: 0.3px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #4a69bd, #1e3799);
        box-shadow: 0 4px 15px rgba(30,55,153,0.4);
        transform: translateY(-1px);
    }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================
if 'telemetry' not in st.session_state:
    st.session_state.telemetry = {
        "model": "GPT-4o",
        "latency": "—",
        "tokens": "—",
        "status": "Ready",
        "timestamp": datetime.now().strftime('%H:%M:%S'),
    }
if 'uploaded_file_path' not in st.session_state:
    st.session_state.uploaded_file_path = None
if 'auto_diagnosis_done' not in st.session_state:
    st.session_state.auto_diagnosis_done = {}

# ============================================================
# PATIENT METADATA
# ============================================================
PATIENT_DATABASE = {
    "P001": {
        "name": "Ahmed Al-Rashid",
        "dob": "1958-03-14",
        "age": 67,
        "gender": "Male",
        "blood_type": "A+",
        "room": "ICU-201",
        "admission": "2025-02-05",
        "attending": "Dr. Sarah Mitchell",
        "diagnosis": "Sepsis (Suspected)",
        "allergies": "Penicillin",
        "weight": "78 kg",
    },
    "P002": {
        "name": "Maria Santos",
        "dob": "1972-07-22",
        "age": 53,
        "gender": "Female",
        "blood_type": "O-",
        "room": "ICU-205",
        "admission": "2025-02-07",
        "attending": "Dr. James Chen",
        "diagnosis": "Cardiac Arrhythmia",
        "allergies": "None Known",
        "weight": "65 kg",
    },
    "P003": {
        "name": "John Williams",
        "dob": "1965-11-09",
        "age": 60,
        "gender": "Male",
        "blood_type": "B+",
        "room": "ICU-210",
        "admission": "2025-02-06",
        "attending": "Dr. Emily Roberts",
        "diagnosis": "Respiratory Failure",
        "allergies": "Sulfa Drugs",
        "weight": "92 kg",
    },
}

DEFAULT_PATIENT = {
    "name": "Unknown Patient", "dob": "N/A", "age": "N/A",
    "gender": "N/A", "blood_type": "N/A", "room": "N/A",
    "admission": "N/A", "attending": "N/A",
    "diagnosis": "Pending Assessment", "allergies": "N/A", "weight": "N/A",
}


def get_patient_info(patient_id):
    return PATIENT_DATABASE.get(patient_id, DEFAULT_PATIENT)


# ============================================================
# VITAL STATUS HELPER
# ============================================================
def vital_status(name, value):
    rules = {
        "hr":   {"normal": (60, 100), "warning": (50, 130)},
        "temp": {"normal": (36.1, 37.9), "warning": (35.5, 39.0)},
        "spo2": {"normal": (95, 100), "warning": (90, 100)},
        "bps":  {"normal": (90, 140), "warning": (80, 160)},
        "bpd":  {"normal": (60, 90),  "warning": (50, 100)},
    }
    r = rules.get(name)
    if not r:
        return "normal"
    if r["normal"][0] <= value <= r["normal"][1]:
        return "normal"
    elif r["warning"][0] <= value <= r["warning"][1]:
        return "warning"
    return "critical"


# ============================================================
# AGENTIC RAG ENGINE WITH OPENAI / GPT-4o
# FIX: Direct OpenAI client to avoid langchain proxies bug
# ============================================================
def run_clinical_agent(file_path, query):
    if not OPENAI_API_KEY:
        return "⚠️ OPENAI_API_KEY not found. Please set it in your .env file."
    try:
        # --- FIX for proxies error ---
        # Try langchain-openai first; if it throws proxies error, fall back
        llm = None
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model="gpt-4o",
                temperature=0,
                api_key=OPENAI_API_KEY,
                max_tokens=4096,
            )
            # Quick test to see if it actually works
            llm.invoke("test")
        except TypeError as te:
            if "proxies" in str(te):
                llm = None
            else:
                raise
        except Exception:
            llm = None

        # Fallback: use langchain_community or older ChatOpenAI
        if llm is None:
            try:
                from langchain_community.chat_models import ChatOpenAI as CommunityChatOpenAI
                llm = CommunityChatOpenAI(
                    model="gpt-4o",
                    temperature=0,
                    openai_api_key=OPENAI_API_KEY,
                    max_tokens=4096,
                )
            except Exception:
                # Last resort: langchain core ChatOpenAI
                from langchain.chat_models import ChatOpenAI as LCChatOpenAI
                llm = LCChatOpenAI(
                    model_name="gpt-4o",
                    temperature=0,
                    openai_api_key=OPENAI_API_KEY,
                    max_tokens=4096,
                )

        agent = create_csv_agent(
            llm, file_path, verbose=False,
            agent_type="openai-functions",
        )
        start_time = time.time()
        token_count = 0
        try:
            response = agent.invoke(query)
            if isinstance(response, dict):
                response = response.get("output", str(response))
        except Exception as e:
            response = f"⚠️ Agent Error: {e}"
        latency = round(time.time() - start_time, 2)

        # Estimate tokens (rough: 1 token ~ 4 chars)
        token_count = len(str(query) + str(response)) // 4

        st.session_state.telemetry = {
            "model": "GPT-4o",
            "latency": f"{latency}s",
            "tokens": f"~{token_count}",
            "status": "Active",
            "timestamp": datetime.now().strftime('%H:%M:%S'),
        }
        return response
    except Exception as e:
        return f"❌ System Error: {e}"


# ============================================================
# CSV VALIDATION
# ============================================================
REQUIRED_COLS = [
    "patient_id", "timestamp", "ECG", "heart_rate_bpm",
    "temperature_c", "bp_systolic_mmHg", "bp_diastolic_mmHg", "spo2_percent",
]

def validate_csv(df):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        return False, f"Missing columns: {', '.join(missing)}"
    return True, "✅ Valid"


# ============================================================
# RENDER FUNCTIONS
# ============================================================
def render_header():
    now = datetime.now().strftime("%b %d, %Y  %H:%M")
    st.markdown(f"""
    <div class="dashboard-header">
        <div class="title-section">
            <h1>🏥 AI ICU Patient Monitor</h1>
            <p>Clinical Decision Support System &nbsp;|&nbsp; {now}</p>
        </div>
        <div>
            <span class="header-badge">● SYSTEM ONLINE</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_patient_banner(patient_id, info):
    initials = "".join([w[0] for w in info["name"].split() if w])[:2].upper()
    st.markdown(f"""
    <div class="patient-banner">
        <div class="patient-avatar">{initials}</div>
        <div class="info-group">
            <span class="info-label">Patient Name</span>
            <span class="info-value">{info['name']}</span>
        </div>
        <div class="info-group">
            <span class="info-label">Patient ID</span>
            <span class="info-value">{patient_id}</span>
        </div>
        <div class="info-group">
            <span class="info-label">Date of Birth</span>
            <span class="info-value">{info['dob']}</span>
        </div>
        <div class="info-group">
            <span class="info-label">Age / Gender</span>
            <span class="info-value">{info['age']} yrs / {info['gender']}</span>
        </div>
        <div class="info-group">
            <span class="info-label">Blood Type</span>
            <span class="info-value">{info['blood_type']}</span>
        </div>
        <div class="info-group">
            <span class="info-label">Room</span>
            <span class="info-value">{info['room']}</span>
        </div>
        <div class="info-group">
            <span class="info-label">Attending</span>
            <span class="info-value">{info['attending']}</span>
        </div>
        <div class="info-group">
            <span class="info-label">Admission</span>
            <span class="info-value">{info['admission']}</span>
        </div>
        <div class="info-group">
            <span class="info-label">Allergies</span>
            <span class="info-value" style="color:#d63031;">{info['allergies']}</span>
        </div>
        <div class="info-group">
            <span class="info-label">Weight</span>
            <span class="info-value">{info['weight']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_vital_card(icon, label, value, unit, normal_range, status):
    dot_class = {"normal": "green", "warning": "yellow", "critical": "red"}[status]
    st.markdown(f"""
    <div class="vital-card {status}">
        <div class="vital-icon">{icon}</div>
        <div class="vital-label">{label}</div>
        <div class="vital-value">{value} <span class="vital-unit">{unit}</span></div>
        <div class="vital-range">
            <span class="status-dot {dot_class}"></span>{normal_range}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_trend_charts(df):
    st.markdown('<div class="section-header">📈 Vital Sign Trends (60-Minute Window)</div>', unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['heart_rate_bpm'],
        name='Heart Rate (BPM)', mode='lines',
        line=dict(color='#e74c3c', width=2),
    ))
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['spo2_percent'],
        name='SpO₂ (%)', mode='lines',
        line=dict(color='#0984e3', width=2),
    ))
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['temperature_c'],
        name='Temp (°C)', mode='lines',
        line=dict(color='#fdcb6e', width=2),
        yaxis='y2',
    ))
    fig.update_layout(
        template='plotly_white', height=340,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(title="BPM / %"),
        yaxis2=dict(title="°C", overlaying='y', side='right'),
        hovermode='x unified',
    )
    st.plotly_chart(fig, width='stretch')

    col1, col2 = st.columns(2)
    with col1:
        fig_bp = go.Figure()
        fig_bp.add_trace(go.Scatter(
            x=df['timestamp'], y=df['bp_systolic_mmHg'],
            name='Systolic', mode='lines+markers',
            line=dict(color='#d63031', width=2), marker=dict(size=3),
        ))
        fig_bp.add_trace(go.Scatter(
            x=df['timestamp'], y=df['bp_diastolic_mmHg'],
            name='Diastolic', mode='lines+markers',
            line=dict(color='#6c5ce7', width=2), marker=dict(size=3),
        ))
        fig_bp.add_hrect(y0=90, y1=140, fillcolor="green", opacity=0.05, line_width=0)
        fig_bp.update_layout(
            title="Blood Pressure Trend", template='plotly_white', height=280,
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis=dict(title="mmHg"),
        )
        st.plotly_chart(fig_bp, width='stretch')

    with col2:
        ecg_counts = df['ECG'].value_counts()
        colors = ['#00b894', '#fdcb6e', '#d63031', '#6c5ce7', '#e17055']
        fig_ecg = go.Figure(data=[go.Pie(
            labels=ecg_counts.index, values=ecg_counts.values,
            hole=0.45, marker=dict(colors=colors[:len(ecg_counts)]),
        )])
        fig_ecg.update_layout(
            title="ECG Rhythm Distribution", template='plotly_white', height=280,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_ecg, width='stretch')


def render_sidebar_telemetry():
    """Render telemetry info in the sidebar."""
    t = st.session_state.telemetry
    st.markdown(f"""
    <div class="sidebar-telemetry">
        <div class="tele-row"><span class="tele-label">Model</span> <span class="tele-value">{t['model']}</span></div>
        <div class="tele-row"><span class="tele-label">Latency</span> <span class="tele-value">{t['latency']}</span></div>
        <div class="tele-row"><span class="tele-label">Tokens</span> <span class="tele-value">{t['tokens']}</span></div>
        <div class="tele-row"><span class="tele-label">Status</span> <span class="tele-value">{t['status']}</span></div>
        <div class="tele-row"><span class="tele-label">Last Run</span> <span class="tele-value">{t['timestamp']}</span></div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### 📂 Data Source")
    data_mode = st.radio("Select Mode:", ["📋 Use Sample Data", "📤 Upload Patient CSV"], label_visibility="collapsed")

    file_path = None

    if data_mode == "📋 Use Sample Data":
        sample_files = {
            "P001 – Sepsis Deterioration": "patient_1_sepsis.csv",
            "P002 – Cardiac Arrhythmia": "patient_2_arrhythmia.csv",
            "P003 – Respiratory Failure": "patient_3_respiratory.csv",
        }
        selection = st.selectbox("Choose Patient:", list(sample_files.keys()))
        fname = sample_files[selection]
        if os.path.exists(fname):
            file_path = fname
        else:
            st.warning(f"⚠️ `{fname}` not found. Run `python patient_csv_gen.py` first.")
    else:
        uploaded = st.file_uploader("Upload Patient CSV", type=["csv"])
        if uploaded:
            temp_path = os.path.join("temp_uploads", uploaded.name)
            os.makedirs("temp_uploads", exist_ok=True)
            with open(temp_path, "wb") as f:
                f.write(uploaded.getbuffer())
            file_path = temp_path
            st.success(f"✅ Uploaded: {uploaded.name}")

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    auto_diagnose = st.toggle("Auto-Run AI Diagnosis", value=True)
    show_raw = st.toggle("Show Raw Data Table", value=False)

    st.markdown("---")
    st.markdown("### 📡 AI Observability")
    render_sidebar_telemetry()

    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; font-size:0.8em; color:#636e72 !important;'>"
        "AI ICU Monitor v2.0<br><b style='color:#1e3799 !important;'>Powered by GPT-4o (OpenAI)</b></p>",
        unsafe_allow_html=True,
    )

# ============================================================
# MAIN DASHBOARD
# ============================================================
render_header()

if file_path is None:
    st.info("👈 Select a patient or upload a CSV from the sidebar to begin monitoring.")
    st.stop()

# Load & validate
df = pd.read_csv(file_path)
valid, msg = validate_csv(df)
if not valid:
    st.error(f"❌ CSV Validation Failed: {msg}")
    st.info(f"Required columns: `{', '.join(REQUIRED_COLS)}`")
    st.stop()

patient_id = str(df['patient_id'].iloc[0])
info = get_patient_info(patient_id)
last = df.iloc[-1]

# Patient Banner
render_patient_banner(patient_id, info)

# Latest Vitals
hr   = float(last['heart_rate_bpm'])
temp = float(last['temperature_c'])
spo2 = float(last['spo2_percent'])
bps  = float(last['bp_systolic_mmHg'])
bpd  = float(last['bp_diastolic_mmHg'])
ecg  = str(last['ECG'])

hr_s   = vital_status("hr", hr)
temp_s = vital_status("temp", temp)
spo2_s = vital_status("spo2", spo2)
bps_s  = vital_status("bps", bps)
bpd_s  = vital_status("bpd", bpd)

is_critical = (
    spo2 < 90 or hr > 150 or temp > 39.5 or bps < 90
    or "V-TACH" in ecg.upper() or "VTACH" in ecg.upper()
)

if is_critical:
    st.markdown(
        '<div class="critical-alert-banner">'
        '🚨 &nbsp; CRITICAL ALERT — IMMEDIATE ATTENTION REQUIRED &nbsp; 🚨'
        '</div>',
        unsafe_allow_html=True,
    )

# Vital Sign Cards (compact)
st.markdown('<div class="section-header">🫀 Current Vital Signs</div>', unsafe_allow_html=True)

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    render_vital_card("❤️", "Heart Rate", f"{hr:.0f}", "BPM", "Normal: 60-100", hr_s)
with c2:
    render_vital_card("🌡️", "Temperature", f"{temp:.1f}", "°C", "Normal: 36.1-37.9", temp_s)
with c3:
    render_vital_card("🫁", "SpO₂", f"{spo2:.0f}", "%", "Normal: ≥95", spo2_s)
with c4:
    render_vital_card("🩸", "BP Systolic", f"{bps:.0f}", "mmHg", "Normal: 90-140", bps_s)
with c5:
    render_vital_card("🩸", "BP Diastolic", f"{bpd:.0f}", "mmHg", "Normal: 60-90", bpd_s)
with c6:
    ecg_s = "critical" if "V-TACH" in ecg.upper() else "normal"
    # Shorten ECG text for card display
    ecg_short = ecg.split("(")[0].strip()
    ecg_abbrevs = {
        "Normal Sinus Rhythm": "NSR",
        "Sinus Tachycardia": "Sinus Tachy",
        "Sinus Bradycardia": "Sinus Brady",
        "Atrial Fibrillation": "A-Fib",
        "Atrial Flutter": "A-Flutter",
        "V-Tach": "V-Tach",
    }
    for full, short in ecg_abbrevs.items():
        if full.lower() in ecg_short.lower():
            ecg_short = short
            break
    if len(ecg_short) > 12:
        ecg_short = ecg_short[:11] + "…"
    render_vital_card("📟", "ECG", ecg_short, "", "", ecg_s)

# ---- AUTO AI DIAGNOSIS (runs automatically on patient load) ----
st.markdown('<div class="section-header">🤖 AI Clinical Decision Support (Agentic RAG)</div>', unsafe_allow_html=True)

AUTO_QUERY = (
    "Analyze all vitals, identify any critical patterns, provide a clinical diagnosis "
    "and recommend immediate nursing interventions. Include specific next steps."
)

if auto_diagnose and file_path:
    # Only auto-run once per patient to avoid infinite reruns
    cache_key = f"{file_path}_{patient_id}"
    if cache_key not in st.session_state.auto_diagnosis_done:
        with st.spinner("🧠 AI Agent automatically analyzing patient data..."):
            clinical_prompt = (
                f"You are a senior ICU clinical decision support AI. "
                f"Patient: {info['name']}, {info['age']}yrs {info['gender']}, "
                f"Blood Type: {info['blood_type']}, Allergies: {info['allergies']}. "
                f"Primary Diagnosis: {info['diagnosis']}.\n\n"
                f"{AUTO_QUERY}\n\n"
                f"Format your response clearly with these sections:\n"
                f"VITAL SIGNS SUMMARY: (brief overview of current vitals and trends)\n"
                f"CLINICAL DIAGNOSIS: (your assessment)\n"
                f"RISK LEVEL: (LOW / MODERATE / HIGH / CRITICAL)\n"
                f"IMMEDIATE NURSING INTERVENTIONS: (numbered list of actions)\n"
                f"NEXT STEPS & PHYSICIAN ORDERS: (what to do next)\n"
            )
            auto_response = run_clinical_agent(file_path, clinical_prompt)
            st.session_state.auto_diagnosis_done[cache_key] = auto_response

    if cache_key in st.session_state.auto_diagnosis_done:
        st.markdown(
            f'<div class="ai-response-box">{st.session_state.auto_diagnosis_done[cache_key]}</div>',
            unsafe_allow_html=True,
        )

# ---- Manual query input ----
st.markdown('<div class="section-header">💬 Ask AI Agent</div>', unsafe_allow_html=True)

col_a, col_b = st.columns([3, 1])
with col_a:
    query = st.text_input(
        "Ask the AI Agent:",
        value="",
        label_visibility="collapsed",
        placeholder="Type a clinical question about this patient...",
    )
with col_b:
    run_btn = st.button("🔍 Run Query", width='stretch')

if run_btn and query.strip():
    with st.spinner("🧠 AI Agent analyzing..."):
        clinical_prompt = (
            f"You are a senior ICU clinical decision support AI. "
            f"Patient: {info['name']}, {info['age']}yrs {info['gender']}, "
            f"Blood Type: {info['blood_type']}, Allergies: {info['allergies']}. "
            f"Primary Diagnosis: {info['diagnosis']}.\n\n"
            f"Analyze this CSV data and answer: {query}"
        )
        response = run_clinical_agent(file_path, clinical_prompt)
    st.markdown(f'<div class="ai-response-box">{response}</div>', unsafe_allow_html=True)

# Trend Charts
render_trend_charts(df)

# Raw Data
if show_raw:
    st.markdown('<div class="section-header">📄 Raw Patient Data</div>', unsafe_allow_html=True)
    st.dataframe(df, width='stretch', height=300)

# Quick Action Buttons
st.markdown('<div class="section-header">⚡ Quick Actions</div>', unsafe_allow_html=True)
qa1, qa2, qa3, qa4 = st.columns(4)
with qa1:
    if st.button("📊 Full Trend Analysis", width='stretch'):
        with st.spinner("Analyzing..."):
            r = run_clinical_agent(file_path, "Provide a comprehensive 60-minute trend analysis of all vital signs. Identify any deterioration patterns.")
        st.markdown(f'<div class="ai-response-box">{r}</div>', unsafe_allow_html=True)
with qa2:
    if st.button("⚠️ Risk Assessment", width='stretch'):
        with st.spinner("Assessing..."):
            r = run_clinical_agent(file_path, "Perform a risk stratification. Rate the patient risk as LOW, MODERATE, HIGH, or CRITICAL with clinical justification.")
        st.markdown(f'<div class="ai-response-box">{r}</div>', unsafe_allow_html=True)
with qa3:
    if st.button("💊 Medication Review", width='stretch'):
        with st.spinner("Reviewing..."):
            r = run_clinical_agent(file_path, "Based on vitals, suggest medication interventions. Note the patient has these allergies: " + info['allergies'])
        st.markdown(f'<div class="ai-response-box">{r}</div>', unsafe_allow_html=True)
with qa4:
    if st.button("📋 Shift Handoff (SBAR)", width='stretch'):
        with st.spinner("Generating..."):
            r = run_clinical_agent(file_path, "Generate a concise shift handoff summary (SBAR format) for the incoming nurse covering this patient.")
        st.markdown(f'<div class="ai-response-box">{r}</div>', unsafe_allow_html=True)
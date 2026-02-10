import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import plotly.graph_objects as go
from datetime import datetime
import requests
import json
import urllib3
from threading import Thread

# Suppress SSL warnings for Splunk self-signed certs
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

try:
    from dotenv import load_dotenv
    from langchain_experimental.agents import create_csv_agent
except ImportError:
    st.error("❌ Missing packages. Run: pip install langchain-experimental python-dotenv tabulate plotly openai langchain-openai requests")
    st.stop()

# --- LOAD ENVIRONMENT ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SPLUNK_HEC_URL = os.getenv("SPLUNK_HEC_URL", "").strip()
SPLUNK_HEC_TOKEN = os.getenv("SPLUNK_HEC_TOKEN", "").strip()

# ============================================================
# SPLUNK HEC HANDLER — FIXED
# ============================================================
class SplunkHandler:
    """
    Sends events to Splunk HTTP Event Collector (HEC).
    
    .env should have:
      SPLUNK_HEC_URL=https://<your-splunk>:8088
      SPLUNK_HEC_TOKEN=<your-hec-token>
    
    The handler auto-appends /services/collector/event if needed.
    """

    def __init__(self, base_url, token):
        self.token = token
        self.connected = False
        self.last_error = None
        self.last_status = None
        self.event_count = 0

        # --- Fix URL: ensure it ends with /services/collector/event ---
        if not base_url or not token:
            self.url = None
            self.last_error = "SPLUNK_HEC_URL or SPLUNK_HEC_TOKEN not set in .env"
            return

        base_url = base_url.rstrip("/")
        # Always normalize to the correct HEC endpoint
        # Strip any partial path and rebuild correctly
        if "/services/collector/event" in base_url:
            # Already correct
            self.url = base_url
        elif "/services/collector" in base_url:
            # Has /services/collector but missing /event
            self.url = base_url.rstrip("/") + "/event"
        else:
            # Just the base URL like https://host:8088
            self.url = f"{base_url}/services/collector/event"

        self.headers = {
            "Authorization": f"Splunk {token}",
            "Content-Type": "application/json",
        }

    def test_connection(self):
        """Synchronous test — call once at startup to verify HEC is reachable."""
        if not self.url:
            return False, self.last_error or "URL not configured"
        try:
            payload = {
                "sourcetype": "ai_monitor",
                "index": "main",
                "event": {
                    "type": "connection_test",
                    "message": "ICU Monitor connected",
                    "timestamp": datetime.now().isoformat(),
                }
            }
            resp = requests.post(
                self.url, headers=self.headers,
                data=json.dumps(payload),
                verify=False, timeout=5
            )
            self.last_status = resp.status_code
            if resp.status_code == 200:
                self.connected = True
                self.last_error = None
                return True, "Connected"
            else:
                self.connected = False
                try:
                    body = resp.json()
                    self.last_error = f"HTTP {resp.status_code}: {body.get('text', resp.text[:100])}"
                except Exception:
                    self.last_error = f"HTTP {resp.status_code}: {resp.text[:100]}"
                return False, self.last_error
        except requests.exceptions.ConnectionError:
            self.connected = False
            self.last_error = f"Connection refused — is Splunk running at {self.url}?"
            return False, self.last_error
        except requests.exceptions.Timeout:
            self.connected = False
            self.last_error = "Timeout — Splunk HEC not responding within 5s"
            return False, self.last_error
        except Exception as e:
            self.connected = False
            self.last_error = str(e)
            return False, self.last_error

    def send(self, event_type, data):
        """Send event to Splunk HEC asynchronously (non-blocking)."""
        if not self.url or not self.token:
            return

        payload = {
            "sourcetype": "ai_monitor",
            "index": "main",
            "event": {
                "type": event_type,
                "app": "icu_monitor",
                "timestamp": datetime.now().isoformat(),
                **data,
            }
        }

        def _post():
            try:
                resp = requests.post(
                    self.url, headers=self.headers,
                    data=json.dumps(payload),
                    verify=False, timeout=3
                )
                self.last_status = resp.status_code
                if resp.status_code == 200:
                    self.event_count += 1
                    self.connected = True
                else:
                    self.connected = False
                    try:
                        self.last_error = f"HTTP {resp.status_code}: {resp.json().get('text','')}"
                    except Exception:
                        self.last_error = f"HTTP {resp.status_code}"
            except Exception as e:
                self.connected = False
                self.last_error = str(e)

        Thread(target=_post, daemon=True).start()


# --- Initialize Splunk & test connection ---
splunk = SplunkHandler(SPLUNK_HEC_URL, SPLUNK_HEC_TOKEN)

if 'splunk_tested' not in st.session_state:
    ok, msg = splunk.test_connection()
    st.session_state.splunk_tested = True
    st.session_state.splunk_ok = ok
    st.session_state.splunk_msg = msg

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI ICU Patient Monitor", page_icon="🏥", layout="wide", initial_sidebar_state="expanded")

# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
.stApp { background: #eef1f5; }
.block-container { padding-top: 0.8rem; padding-bottom: 0; max-width: 1400px; }
#MainMenu, footer, header { visibility: hidden; }

section[data-testid="stSidebar"] { background: #f7f8fa !important; border-right: 1px solid #e0e4ea; }
section[data-testid="stSidebar"] * { color: #333 !important; }
section[data-testid="stSidebar"] h3 { color: #1a1a2e !important; font-size: 0.95em !important; }
section[data-testid="stSidebar"] hr { border-color: #e0e4ea !important; }

.tele-box { background: #f0f2f5; border: 1px solid #dde1e7; border-radius: 8px; padding: 8px 12px; font-size: 0.73em; margin-top: 6px; }
.tele-box table { width: 100%; border-collapse: collapse; }
.tele-box td { padding: 3px 0; vertical-align: top; }
.tele-box .tk { color: #888; font-weight: 600; text-transform: uppercase; font-size: 0.88em; letter-spacing: 0.3px; }
.tele-box .tv { color: #1a1a2e; font-weight: 700; text-align: right; font-family: 'SF Mono','Consolas',monospace; }

.splunk-status { padding: 6px 10px; border-radius: 6px; font-size: 0.73em; margin-top: 6px; }
.splunk-ok { background: #ecfdf5; border: 1px solid #a7f3d0; color: #065f46; }
.splunk-fail { background: #fef2f2; border: 1px solid #fecaca; color: #991b1b; }

.hdr { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); padding: 14px 24px; border-radius: 10px; margin-bottom: 16px; display: flex; align-items: center; justify-content: space-between; box-shadow: 0 3px 12px rgba(0,0,0,0.12); }
.hdr h1 { margin:0; font-size:1.35em; font-weight:700; color:#fff; letter-spacing:0.3px; }
.hdr .sub { font-size:0.78em; color:#94a3b8; margin-top:1px; }
.hdr .badge { background:#10b981; color:#fff; padding:5px 14px; border-radius:16px; font-size:0.72em; font-weight:600; }

.pt-banner { background: #fff; border-radius: 10px; padding: 14px 20px; margin-bottom: 16px; box-shadow: 0 1px 6px rgba(0,0,0,0.04); border-left: 4px solid #3b82f6; display: flex; flex-wrap: wrap; gap: 24px; align-items: center; }
.pt-banner .avatar { width: 46px; height: 46px; border-radius: 50%; background: linear-gradient(135deg, #3b82f6, #8b5cf6); display: flex; align-items: center; justify-content: center; font-size: 1.1em; color: #fff; font-weight: 700; flex-shrink: 0; }
.pt-banner .fg { display: flex; flex-direction: column; }
.pt-banner .fl { font-size: 0.6em; color: #94a3b8; text-transform: uppercase; font-weight: 600; letter-spacing: 0.6px; }
.pt-banner .fv { font-size: 0.88em; color: #1e293b; font-weight: 600; }
.pt-banner .fv.red { color: #ef4444; }

.vc-row { display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px; margin-bottom: 18px; }
.vc { background: #fff; border-radius: 10px; padding: 14px 10px; text-align: center; box-shadow: 0 1px 6px rgba(0,0,0,0.04); border-left: 4px solid #e2e8f0; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 110px; }
.vc .vi { font-size: 1.0em; margin-bottom: 2px; }
.vc .vl { font-size: 0.58em; color: #94a3b8; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px; margin-bottom: 4px; }
.vc .vv { font-size: 1.55em; font-weight: 800; line-height: 1.1; white-space: nowrap; }
.vc .vu { font-size: 0.55em; color: #94a3b8; font-weight: 500; }
.vc .vr { font-size: 0.55em; color: #1e293b; margin-top: 4px; }
.vc.ok { border-left-color: #10b981; } .vc.ok .vv { color: #059669; }
.vc.warn { border-left-color: #f59e0b; } .vc.warn .vv { color: #d97706; }
.vc.crit { border-left-color: #ef4444; } .vc.crit .vv { color: #dc2626; }

.crit-alert { background: #fef2f2; border: 1px solid #fecaca; border-left: 5px solid #ef4444; color: #991b1b; padding: 12px 20px; border-radius: 8px; font-weight: 700; font-size: 0.92em; margin-bottom: 16px; display: flex; align-items: center; gap: 10px; }
.sec { font-size: 0.95em; font-weight: 700; color: #1e293b; margin: 20px 0 10px 0; padding-bottom: 6px; border-bottom: 2px solid #e2e8f0; }
.ai-box { background: #fff; border: 1px solid #e2e8f0; border-left: 4px solid #3b82f6; border-radius: 8px; padding: 16px 20px; margin: 12px 0; font-size: 0.88em; line-height: 1.75; color: #334155; max-height: 420px; overflow-y: auto; box-shadow: 0 1px 4px rgba(0,0,0,0.03); }
.ai-box b, .ai-box strong { color: #1e293b; }

.stButton > button { background: #1e293b !important; color: #fff !important; border: none !important; border-radius: 6px !important; padding: 6px 16px !important; font-weight: 600 !important; font-size: 0.82em !important; transition: all 0.2s; }
.stButton > button:hover { background: #334155 !important; box-shadow: 0 2px 8px rgba(0,0,0,0.15); }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================
if 'telemetry' not in st.session_state:
    st.session_state.telemetry = {"model":"GPT-4o","latency":"—","tokens":"—","status":"Ready","ts":datetime.now().strftime('%H:%M:%S')}
if 'auto_dx' not in st.session_state:
    st.session_state.auto_dx = {}

# ============================================================
# PATIENT DATABASE
# ============================================================
PATIENTS = {
    "P001": dict(name="Ahmed Al-Rashid", dob="1958-03-14", age=67, gender="Male", blood="A+", room="ICU-201",
                 admission="2025-02-05", attending="Dr. Sarah Mitchell", dx="Sepsis (Suspected)", allergy="Penicillin", weight="78 kg"),
    "P002": dict(name="Maria Santos", dob="1972-07-22", age=53, gender="Female", blood="O-", room="ICU-205",
                 admission="2025-02-07", attending="Dr. James Chen", dx="Cardiac Arrhythmia", allergy="None Known", weight="65 kg"),
    "P003": dict(name="John Williams", dob="1965-11-09", age=60, gender="Male", blood="B+", room="ICU-210",
                 admission="2025-02-06", attending="Dr. Emily Roberts", dx="Respiratory Failure", allergy="Sulfa Drugs", weight="92 kg"),
}
DEFAULT_PT = dict(name="Unknown", dob="N/A", age="N/A", gender="N/A", blood="N/A", room="N/A",
                  admission="N/A", attending="N/A", dx="Pending", allergy="N/A", weight="N/A")

# ============================================================
# HELPERS
# ============================================================
def vstat(key, val):
    R = {"hr":(60,100,50,130),"temp":(36.1,37.9,35.5,39.0),"spo2":(95,100,90,100),"bps":(90,140,80,160),"bpd":(60,90,50,100)}
    r = R.get(key)
    if not r: return "ok"
    if r[0] <= val <= r[1]: return "ok"
    if r[2] <= val <= r[3]: return "warn"
    return "crit"

def ecg_short(ecg):
    m = {"Normal Sinus Rhythm":"NSR","Sinus Tachycardia":"S-Tachy","Sinus Bradycardia":"S-Brady",
         "Atrial Fibrillation":"A-Fib","Atrial Flutter":"A-Flutter"}
    e = ecg.split("(")[0].strip()
    for k,v in m.items():
        if k.lower() in e.lower(): return v
    return e[:10] if len(e)>10 else e

REQUIRED = ["patient_id","timestamp","ECG","heart_rate_bpm","temperature_c","bp_systolic_mmHg","bp_diastolic_mmHg","spo2_percent"]

def validate(df):
    miss = [c for c in REQUIRED if c not in df.columns]
    return (True,"") if not miss else (False,f"Missing: {', '.join(miss)}")

# ============================================================
# AI AGENT — with Splunk logging
# ============================================================
def run_agent(fp, query):
    if not OPENAI_API_KEY:
        return "⚠️ OPENAI_API_KEY not set in .env"
    try:
        llm = None
        for attempt in range(3):
            try:
                if attempt == 0:
                    from langchain_openai import ChatOpenAI
                    llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_API_KEY, max_tokens=4096)
                elif attempt == 1:
                    from langchain_community.chat_models import ChatOpenAI as C2
                    llm = C2(model="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY, max_tokens=4096)
                else:
                    from langchain.chat_models import ChatOpenAI as C3
                    llm = C3(model_name="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY, max_tokens=4096)
                break
            except (TypeError, ImportError):
                llm = None
                continue
        if llm is None:
            return "❌ Could not initialise LLM. Run: pip install --upgrade openai langchain-openai"

        agent = create_csv_agent(llm, fp, verbose=False, agent_type="openai-functions", allow_dangerous_code=True)
        t0 = time.time()
        resp = agent.invoke(query)
        if isinstance(resp, dict): resp = resp.get("output", str(resp))

        lat = round(time.time() - t0, 2)
        toks = len(str(query) + str(resp)) // 4

        # --- LOG TO SPLUNK ---
        splunk.send("ai_inference", {
            "model": "gpt-4o",
            "latency_sec": lat,
            "tokens_estimated": toks,
            "query_length": len(query),
            "response_length": len(str(resp)),
            "status": "success",
        })

        st.session_state.telemetry = {
            "model": "GPT-4o",
            "latency": f"{lat}s",
            "tokens": f"~{toks}",
            "status": "Active",
            "ts": datetime.now().strftime('%H:%M:%S'),
            "splunk_events": splunk.event_count,
        }
        return resp
    except Exception as e:
        splunk.send("ai_error", {"error": str(e)})
        return f"❌ Error: {e}"


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### 📂 Data Source")
    mode = st.radio("Mode", ["📋 Sample Data", "📤 Upload CSV"], label_visibility="collapsed")
    fp = None
    if mode == "📋 Sample Data":
        sf = {"P001 – Sepsis":"patient_1_sepsis.csv","P002 – Arrhythmia":"patient_2_arrhythmia.csv","P003 – Resp. Failure":"patient_3_respiratory.csv"}
        sel = st.selectbox("Patient", list(sf.keys()))
        fn = sf[sel]
        fp = fn if os.path.exists(fn) else None
        if fp is None: st.warning(f"⚠️ `{fn}` not found. Run `python patient_csv_gen.py`")
    else:
        up = st.file_uploader("Upload CSV", type=["csv"])
        if up:
            os.makedirs("temp_uploads", exist_ok=True)
            tp = os.path.join("temp_uploads", up.name)
            with open(tp,"wb") as f: f.write(up.getbuffer())
            fp = tp
            st.success(f"✅ {up.name}")

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    auto_dx = st.toggle("Auto AI Diagnosis", value=True)
    show_raw = st.toggle("Show Raw Data", value=False)

    st.markdown("---")
    st.markdown("### 📡 Observability")

    # --- Splunk Status with debug info ---
    if st.session_state.get('splunk_ok'):
        st.markdown(
            f'<div class="splunk-status splunk-ok">✅ <b>Splunk HEC Connected</b><br>'
            f'URL: {splunk.url}<br>Events sent: {splunk.event_count}</div>',
            unsafe_allow_html=True)
    else:
        err = st.session_state.get('splunk_msg', 'Not configured')
        st.markdown(
            f'<div class="splunk-status splunk-fail">❌ <b>Splunk HEC:</b> {err}<br>'
            f'URL: {SPLUNK_HEC_URL or "not set"}</div>',
            unsafe_allow_html=True)

    # Retry button
    if st.button("🔄 Test Splunk", use_container_width=True):
        ok, msg = splunk.test_connection()
        st.session_state.splunk_ok = ok
        st.session_state.splunk_msg = msg
        if ok:
            st.success("✅ Splunk HEC is reachable!")
        else:
            st.error(f"❌ {msg}")

    t = st.session_state.telemetry
    st.markdown(f"""<div class="tele-box"><table>
    <tr><td class="tk">Model</td><td class="tv">{t['model']}</td></tr>
    <tr><td class="tk">Latency</td><td class="tv">{t['latency']}</td></tr>
    <tr><td class="tk">Tokens</td><td class="tv">{t['tokens']}</td></tr>
    <tr><td class="tk">Status</td><td class="tv">{t['status']}</td></tr>
    <tr><td class="tk">Last Run</td><td class="tv">{t['ts']}</td></tr>
    <tr><td class="tk">Splunk Events</td><td class="tv">{splunk.event_count}</td></tr>
    </table></div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<p style='text-align:center;font-size:0.72em;color:#94a3b8;'>ICU Monitor v2.0 · GPT-4o · Splunk HEC</p>", unsafe_allow_html=True)

# ============================================================
# MAIN
# ============================================================
now = datetime.now().strftime("%b %d, %Y · %H:%M")
st.markdown(f"""<div class="hdr">
<div><h1>🏥 AI ICU Patient Monitor</h1><div class="sub">Clinical Decision Support System &nbsp;·&nbsp; {now}</div></div>
<span class="badge">● ONLINE</span></div>""", unsafe_allow_html=True)

if fp is None:
    st.info("👈 Select a patient or upload a CSV to begin.")
    st.stop()

df = pd.read_csv(fp)
ok, msg = validate(df)
if not ok:
    st.error(f"CSV Error: {msg}")
    st.stop()

pid = str(df['patient_id'].iloc[0])
pt = PATIENTS.get(pid, DEFAULT_PT)
last = df.iloc[-1]

hr = float(last['heart_rate_bpm'])
tmp = float(last['temperature_c'])
sp = float(last['spo2_percent'])
bps = float(last['bp_systolic_mmHg'])
bpd = float(last['bp_diastolic_mmHg'])
ecg = str(last['ECG'])

# --- LOG VITALS TO SPLUNK ---
splunk.send("patient_vitals", {
    "patient_id": pid,
    "patient_name": pt['name'],
    "heart_rate_bpm": hr,
    "spo2_percent": sp,
    "temperature_c": tmp,
    "bp_systolic": bps,
    "bp_diastolic": bpd,
    "ecg": ecg,
    "status": "critical" if (hr > 130 or sp < 90 or tmp > 39.5 or bps < 90) else "stable",
})

# — Patient Banner —
ini = "".join([w[0] for w in pt["name"].split()])[:2].upper()
st.markdown(f"""<div class="pt-banner">
<div class="avatar">{ini}</div>
<div class="fg"><span class="fl">Patient</span><span class="fv">{pt['name']}</span></div>
<div class="fg"><span class="fl">ID</span><span class="fv">{pid}</span></div>
<div class="fg"><span class="fl">DOB</span><span class="fv">{pt['dob']}</span></div>
<div class="fg"><span class="fl">Age/Sex</span><span class="fv">{pt['age']}y {pt['gender']}</span></div>
<div class="fg"><span class="fl">Blood</span><span class="fv">{pt['blood']}</span></div>
<div class="fg"><span class="fl">Room</span><span class="fv">{pt['room']}</span></div>
<div class="fg"><span class="fl">Attending</span><span class="fv">{pt['attending']}</span></div>
<div class="fg"><span class="fl">Admitted</span><span class="fv">{pt['admission']}</span></div>
<div class="fg"><span class="fl">Allergies</span><span class="fv red">{pt['allergy']}</span></div>
<div class="fg"><span class="fl">Weight</span><span class="fv">{pt['weight']}</span></div>
</div>""", unsafe_allow_html=True)

hs, ts, ss, bss, bds = vstat("hr",hr), vstat("temp",tmp), vstat("spo2",sp), vstat("bps",bps), vstat("bpd",bpd)
es = "crit" if "V-TACH" in ecg.upper() else "ok"
is_crit = sp<90 or hr>150 or tmp>39.5 or bps<90 or "V-TACH" in ecg.upper()

if is_crit:
    st.markdown('<div class="crit-alert">🚨 CRITICAL — Immediate clinical attention required</div>', unsafe_allow_html=True)
    splunk.send("critical_alert", {"patient_id": pid, "patient_name": pt['name'], "ecg": ecg,
        "heart_rate": hr, "spo2": sp, "temperature": tmp, "bp_systolic": bps})

# — Vital Cards —
st.markdown('<div class="sec">🫀 Current Vital Signs</div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="vc-row">
  <div class="vc {hs}"><div class="vi">❤️</div><div class="vl">Heart Rate</div><div class="vv">{hr:.0f} <span class="vu">BPM</span></div><div class="vr">Normal 60–100</div></div>
  <div class="vc {ts}"><div class="vi">🌡️</div><div class="vl">Temperature</div><div class="vv">{tmp:.1f} <span class="vu">°C</span></div><div class="vr">Normal 36.1–37.9</div></div>
  <div class="vc {ss}"><div class="vi">🫁</div><div class="vl">SpO₂</div><div class="vv">{sp:.0f} <span class="vu">%</span></div><div class="vr">Normal ≥95</div></div>
  <div class="vc {bss}"><div class="vi">🩸</div><div class="vl">BP Systolic</div><div class="vv">{bps:.0f} <span class="vu">mmHg</span></div><div class="vr">Normal 90–140</div></div>
  <div class="vc {bds}"><div class="vi">🩸</div><div class="vl">BP Diastolic</div><div class="vv">{bpd:.0f} <span class="vu">mmHg</span></div><div class="vr">Normal 60–90</div></div>
  <div class="vc {es}"><div class="vi">📟</div><div class="vl">ECG</div><div class="vv">{ecg_short(ecg)}</div><div class="vr">&nbsp;</div></div>
</div>
""", unsafe_allow_html=True)

# — Trend Charts —
st.markdown('<div class="sec">📈 Trends (60 min)</div>', unsafe_allow_html=True)
cm = dict(l=8, r=8, t=36, b=8)
fs = dict(family='Arial', size=11, color='#1e293b')

tc1, tc2, tc3 = st.columns(3)
with tc1:
    f1 = go.Figure()
    f1.add_trace(go.Scatter(x=df['timestamp'], y=df['heart_rate_bpm'], name='HR', line=dict(color='#ef4444',width=3), line_shape='spline'))
    f1.add_hrect(y0=60, y1=100, fillcolor="#10b981", opacity=0.06, line_width=0)
    f1.update_layout(title="Heart Rate", template='plotly_white', height=300, margin=cm, yaxis=dict(title="BPM"), hovermode='x unified', font=fs)
    st.plotly_chart(f1, use_container_width=True)
with tc2:
    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=df['timestamp'], y=df['spo2_percent'], name='SpO₂', line=dict(color='#2563eb',width=3), line_shape='spline', fill='tozeroy', fillcolor='rgba(59,130,246,0.06)'))
    f2.add_hrect(y0=95, y1=100, fillcolor="#10b981", opacity=0.06, line_width=0)
    f2.update_layout(title="SpO₂", template='plotly_white', height=300, margin=cm, yaxis=dict(title="%", range=[max(80,df['spo2_percent'].min()-2),100]), hovermode='x unified', font=fs)
    st.plotly_chart(f2, use_container_width=True)
with tc3:
    f3 = go.Figure()
    f3.add_trace(go.Scatter(x=df['timestamp'], y=df['temperature_c'], name='Temp', line=dict(color='#f59e0b',width=3), line_shape='spline'))
    f3.add_hrect(y0=36.1, y1=37.9, fillcolor="#10b981", opacity=0.06, line_width=0)
    f3.update_layout(title="Temperature", template='plotly_white', height=300, margin=cm, yaxis=dict(title="°C"), hovermode='x unified', font=fs)
    st.plotly_chart(f3, use_container_width=True)

# — AI Diagnosis —
st.markdown('<div class="sec">🤖 AI Clinical Decision Support</div>', unsafe_allow_html=True)

AUTO_Q = (
    "Analyze all vitals and trends. Provide a CONCISE clinical assessment in this exact format:\n"
    "SUMMARY: (2-3 sentence overview)\n"
    "DIAGNOSIS: (primary and differential)\n"
    "RISK LEVEL: (LOW/MODERATE/HIGH/CRITICAL)\n"
    "IMMEDIATE ACTIONS: (numbered, max 5 key interventions)\n"
    "NEXT STEPS: (what to monitor and physician orders)\n"
    "Keep the entire response under 250 words. Be direct and clinical."
)

if auto_dx and fp:
    ck = f"{fp}_{pid}"
    if ck not in st.session_state.auto_dx:
        with st.spinner("🧠 Running AI diagnosis…"):
            prompt = (
                f"You are a senior ICU clinical AI. Patient: {pt['name']}, {pt['age']}y {pt['gender']}, "
                f"Blood: {pt['blood']}, Allergies: {pt['allergy']}, Dx: {pt['dx']}.\n\n{AUTO_Q}"
            )
            st.session_state.auto_dx[ck] = run_agent(fp, prompt)
    if ck in st.session_state.auto_dx:
        st.markdown(f'<div class="ai-box">{st.session_state.auto_dx[ck]}</div>', unsafe_allow_html=True)

# — Manual Query —
c1, c2 = st.columns([4,1])
with c1:
    q = st.text_input("query", value="", label_visibility="collapsed", placeholder="Ask a clinical question…")
with c2:
    go_btn = st.button("🔍 Ask AI", use_container_width=True)
if go_btn and q.strip():
    with st.spinner("Analyzing…"):
        r = run_agent(fp, f"Patient: {pt['name']}, {pt['age']}y, Allergies: {pt['allergy']}, Dx: {pt['dx']}. {q}. Keep response concise, under 200 words.")
    st.markdown(f'<div class="ai-box">{r}</div>', unsafe_allow_html=True)

# — BP & ECG Charts —
cc1, cc2 = st.columns(2)
with cc1:
    fbp = go.Figure()
    fbp.add_trace(go.Scatter(x=df['timestamp'], y=df['bp_systolic_mmHg'], name='Systolic', line=dict(color='#ef4444',width=3), line_shape='spline'))
    fbp.add_trace(go.Scatter(x=df['timestamp'], y=df['bp_diastolic_mmHg'], name='Diastolic', line=dict(color='#8b5cf6',width=3), line_shape='spline'))
    fbp.add_hrect(y0=90, y1=140, fillcolor="#10b981", opacity=0.04, line_width=0)
    fbp.update_layout(title="Blood Pressure", template='plotly_white', height=300, margin=cm,
        legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1), yaxis=dict(title="mmHg"), font=fs)
    st.plotly_chart(fbp, use_container_width=True)
with cc2:
    ec = df['ECG'].value_counts()
    fec = go.Figure(data=[go.Pie(labels=ec.index, values=ec.values, hole=0.45,
        marker=dict(colors=['#10b981','#f59e0b','#ef4444','#8b5cf6','#f97316'][:len(ec)], line=dict(color='#fff',width=1)), textinfo='percent+label')])
    fec.update_layout(title="ECG Distribution", template='plotly_white', height=300, margin=cm, font=fs)
    st.plotly_chart(fec, use_container_width=True)

# — Raw Data —
if show_raw:
    st.markdown('<div class="sec">📄 Raw Data</div>', unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True, height=250)

# — Quick Actions —
st.markdown('<div class="sec">⚡ Quick Actions</div>', unsafe_allow_html=True)
b1,b2,b3,b4 = st.columns(4)
with b1:
    if st.button("📊 Trend Analysis", use_container_width=True):
        with st.spinner("…"):
            r = run_agent(fp, "60-minute vital sign trend analysis. Be concise, under 150 words.")
        st.markdown(f'<div class="ai-box">{r}</div>', unsafe_allow_html=True)
with b2:
    if st.button("⚠️ Risk Score", use_container_width=True):
        with st.spinner("…"):
            r = run_agent(fp, "Risk stratification: rate LOW/MODERATE/HIGH/CRITICAL. Give 3 reasons. Under 100 words.")
        st.markdown(f'<div class="ai-box">{r}</div>', unsafe_allow_html=True)
with b3:
    if st.button("💊 Medications", use_container_width=True):
        with st.spinner("…"):
            r = run_agent(fp, f"Suggest medications. Allergies: {pt['allergy']}. Under 120 words.")
        st.markdown(f'<div class="ai-box">{r}</div>', unsafe_allow_html=True)
with b4:
    if st.button("📋 SBAR Handoff", use_container_width=True):
        with st.spinner("…"):
            r = run_agent(fp, "Shift handoff in SBAR format. Under 150 words.")
        st.markdown(f'<div class="ai-box">{r}</div>', unsafe_allow_html=True)
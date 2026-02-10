# 🏥 ICU Patient Monitoring System

## SIMPLE 3-STEP SETUP

### Step 1: Install Requirements
```bash
pip install streamlit pandas numpy python-dotenv langchain-openai langchain-experimental langchain-community openai
```

OR:
```bash
pip install -r requirements.txt
```

### Step 2: Add Your API Key
Create a file called `.env` in the same folder and add:
```
OPENAI_API_KEY=sk-your-actual-api-key-here
```

### Step 3: Run the App
```bash
streamlit run patient_monitor.py
```

**That's it!** The CSV files are already included.

---

## 📁 Files Included

- `patient_monitor.py` - Main application (COMPLETE)
- `patient_csv_gen.py` - Data generator
- `patient_1_sepsis.csv` - Sample patient 1
- `patient_2_arrhythmia.csv` - Sample patient 2
- `patient_3_respiratory.csv` - Sample patient 3
- `requirements.txt` - Dependencies
- `.env.example` - API key template

---

## ✨ Features

- ✅ Real-time vital signs display
- ✅ Temperature in Fahrenheit
- ✅ Critical alert detection
- ✅ AI analysis with GPT-4o
- ✅ Ask questions about patient data
- ✅ Charts and trends
- ✅ 3 sample patients included

---

## 🧪 Test It

1. Run: `streamlit run patient_monitor.py`
2. Select: "P002 - V-Tach Episode"
3. See critical alert around minute 40
4. Ask AI: "What was the maximum heart rate?"

---

## ❓ Troubleshooting

**Can't find CSV files?**
```bash
python patient_csv_gen.py
```

**Missing libraries?**
```bash
pip install -r requirements.txt
```

**No API key?**
Create `.env` file with your OpenAI API key

---

**Ready to run! Simple and clean!** 🚀

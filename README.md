# Scenario-2-Health-Sensing
- This project focuses on analyzing overnight sleep data to detect abnormal breathing patterns such as Hypopnea and Obstructive Apnea.

**Dataset Description**
The dataset consist of five folders(AP01,AP02,...,AP05) with each folder consisting of:
- Flow-.txt → Nasal airflow (32 Hz)
- Thorac-.txt → Thoracic movement (32 Hz)
- SPO2-.txt → Oxygen saturation (4 Hz)
- FlowEvents-.txt → Breathing event annotations
- SleepProfile-.txt → Sleep stage information
- 🔗: https://drive.google.com/drive/folders/1J95cTl574LLdj4uelYwjyv0094d8sOpD?usp=sharing

**Data Visualization**
- vis.py generates plots of Nasal Airflow, Thoracic Movement, SpO₂
- Each participant’s visualization is saved as a PDF file in visualisation folder.

**Data Preprocessing & Windowing**
- Signals are:
  - Time-aligned using timestamps
  - Split into 30-second windows
  - 50% overlap between consecutive windows
 
**Dataset Creation**
- Each window is labeled based on overlap with annotated events Hypopnea, Obstructive Apnea and Normal.

**Modeling & Evaluation**

- Models implemented 1D CNN & 1D Conv-LSTM
- Evaluation strategy **Leave-One-Participant-Out Cross-Validation** (LOPO-CV)

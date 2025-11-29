# Multi-Sensor Fusion for Early Anomaly Detection  
A lightweight AI system that fuses **heart rate**, **SpO₂**, **temperature**, and **motion** signals to detect early physiological anomalies.  
Built using **PyTorch**, **LSTM autoencoders**, and a **rule-based fusion engine**.

This project demonstrates how multi-modal biosignals can be combined to create a robust early-warning system for preventive health monitoring.

---

## Project Overview  
Modern wearables collect multiple streams of physiological data, but each signal alone is noisy.  
The goal of this project is to fuse **4 biosignals** into one unified **early risk score**:

- Heart Rate (HR)
- Blood Oxygen Saturation (SpO₂)
- Skin Temperature
- Motion / Activity

We implement two approaches:

### **1. Weighted Rule-Based Fusion**
- Normalize each signal (z-score)
- Apply clinically-informed weights
- Smooth for stability
- Flag anomalies when fusion score > 2 SD from baseline

### **2. LSTM Autoencoder Fusion Model**
A sequence-to-sequence LSTM learns normal multi-sensor patterns and reconstructs them.  
High reconstruction error = anomaly.

---

## Repository Structure
```python
multisensor-fusion-health/
│
├── data/
│ └── sample_multisensor.csv
│
├── notebooks/
│ └── multisensor_demo.ipynb
│
├── src/
│ ├── preprocess.py
│ ├── fuse.py
│ ├── model.py
│ ├── visualize.py
│
├── model/
│ └── fusion_lstm.pth
│
├── README.md
└── requirements.txt
```
---

## Signals Included

| Sensor        | Description                                     |
|---------------|-------------------------------------------------|
| Heart Rate    | Synthetic BPM based on resting + spikes         |
| SpO₂          | Baseline 96–99% with occasional dips            |
| Temperature   | Low-variance 36.4–37°C with fever spikes        |
| Motion        | Low baseline with bursts of activity            |

Dataset length: **10,800 samples** (3 hours @ 1 Hz)

---

## Fusion Pipeline

### **1. Z-score normalization**
```python
df[f"{s}_z"] = (df[s] - df[s].mean()) / df[s].std()
```

### **2. Weighted fusion**

Weights:
- HR: 0.35
- SpO₂: 0.35
- Temperature: 0.15
- Motion: 0.15

```python
df["fusion_score"] = (
    df["heart_rate_z"]*0.35 +
    df["spo2_z"]*0.35 +
    df["temperature_z"]*0.15 +
    df["motion_z"]*0.15)
```
### **3. Smoothing (rolling mean)**
```python
df["fusion_smooth"] = df["fusion_score"].rolling(60).mean()
```

### **4. Anomaly threshold**
```python
threshold = df["fusion_smooth"].mean() + 2*df["fusion_smooth"].std()
df["anomaly"] = df["fusion_smooth"] > threshold
```
### **LSTM Autoencoder**

Architecture
- Input: 60-second window of 4 signals
- Encoder: LSTM(hidden=16)
- Decoder: LSTM(hidden=4)
- Output: reconstruction of original sequence
- Loss: MSE

```python
class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features=4, embedding_dim=16):
        super().__init__()
        self.encoder = nn.LSTM(n_features, embedding_dim, batch_first=True)
        self.decoder = nn.LSTM(embedding_dim, n_features, batch_first=True)

    def forward(self, x):
        encoded, _ = self.encoder(x)
        last = encoded[:, -1, :].unsqueeze(1)
        repeated = last.repeat(1, x.size(1), 1)
        decoded, _ = self.decoder(repeated)
        return decoded
```
### **Training (10 epochs)**
Loss decreased smoothly over time.
```python
Reconstruction Error
df_lstm["recon_error"] = errors
df_lstm["lstm_anomaly"] = df_lstm["recon_error"] > threshold_lstm
```
### **Visualizations**
**1. Raw Signals (HR, SpO₂, Temp, Motion)**
4-panel time series plots showing synthetic physiology with injected anomalies.

**2. Fused Risk Score**
Shows how the weighted fusion reacts to multi-modal anomalies.

**3. LSTM Reconstruction Error Plot**
Displays anomaly peaks flagged by the model.

All visualizations are generated inside:
notebooks/multisensor_demo.ipynb

### **Running the Project**
**Install dependencies**
```python
pip install -r requirements.txt
```
### **Run the notebook**
```python
jupyter notebook notebooks/multisensor_demo.ipynb
```
### **Re-train the LSTM model**
python src/model.py

### **Why This Project Matters**
Most wearables analyze signals separately. This project shows how multi-sensor fusion provides:
- better robustness
- fewer false alarms
- early identification of physiological deterioration
- stronger real-world health insights

This is foundational for **preventive healthcare systems**, especially in low-data or low-connectivity settings.

### **Model Card**
- Model: LSTM Autoencoder (4-signal fusion)
- Training Data: Synthetic dataset (10,800 samples)
- Task: Reconstruction-based anomaly detection
- Intended Use: Research, prototyping, signal fusion experiments
- Not for clinical or diagnostic use.
- Limitations:
  - Synthetic data; real sensor noise not fully represented
  - Model not externally validated
  - Interpretability dependent on signal quality

Weights available at:
model/fusion_lstm.pth

### **Future Work**
- Add HRV + respiration
- Transformer-based fusion
- Real wearable data integration
- On-device inference optimization
- Export ONNX/TFLite models
- Expand to multi-day baseline modeling

# Pattern-based Anomaly &amp; Manipulation Monitoring System (PAMMS)


**(Isolation Forest (IF) + Autoencoder/PCA (AE) + Graph-Based Collusion Detection)**

PAMMS is an intelligent framework designed to detect abnormal trading behaviors and potential collusion among traders using real tick-level market data.  
This version **does not generate synthetic data** and only processes the real Excel dataset provided by the user.

---

## 1. Overview

The PAMMS system analyzes trading tick data and identifies suspicious market behaviors using:
- **Schema normalization** (mapping mixed-language Excel columns automatically)
- **Feature engineering** (statistical and behavioral attributes)
- **Anomaly detection** using:
  - Isolation Forest (point anomalies)
  - Autoencoder (or PCA fallback if PyTorch unavailable)
- **Graph-based clustering** to detect possible collusive trader groups
- **Sliding-window event labeling** to identify time-localized manipulation patterns

---

## 2. Libraries and Setup

### Core Libraries Used:
| Component | Library |
|-----------|---------|
| Data Handling | `pandas`, `numpy` |
| Machine Learning | `scikit-learn` (`IsolationForest`, `PCA`, `StandardScaler`, `SpectralClustering`) |
| Graph / Network | `networkx` |
| Visualization | `matplotlib` |
| Optional Autoencoder | `torch` |
| Reading Excel files | `openpyxl` |

### Output Directory:
All logs, charts, and CSV results are automatically saved to:


### Detection Thresholds:
| Setting | Value |
|---------|--------|--------|
| `TH_IF = 0.60` | Minimum anomaly score for Isolation Forest |
| `TH_AE = 0.80` | Minimum anomaly score for Autoencoder/PCA |
| `TOP_K_EVENTS = 900` | Number of top anomalous ticks exported |
| `WINDOW_SEC = 60` | Sliding window duration |
| `STEP_SEC = 25` | Window shift interval |

---

## 3. Data Normalization (`normalize_tick_schema`)

The Excel dataset is automatically mapped to this unified schema:

| Normalized Column | Possible Source Names (auto-detected) |
|------------------|----------------------------------------|
| `Time` | `DateTime`, `date`, `time`, `timestamp`, `تاریخ`, `زمان` |
| `Price` | `price`, `close`, `last`, `قیمت`, `پایانی` |
| `Volume` | `volume`, `vol`, `qty`, `quantity`, `حجم` |
| `Side` | `OrderType`, `side`, `direction`, `خرید/فروش` |
| `TraderID` | `TraderID`, `account`, `client`, `کد معاملاتی` |
| `Symbol` | `symbol`, `ticker`, `نماد` |

`Side` is normalized to:
- `B` → Buy
- `S` → Sell  
*(Fuzzy matching supports Persian / English / numeric values)*

If `TraderID` or `Symbol` are missing, placeholders are generated.

---

## 4. Data Ingestion (`DataIngestor`)

- Reads **only the real Excel file**
- Does **not** create synthetic data under any conditions
- If the Excel file is missing → script raises an error

Default expected file:


---

## 5. Feature Engineering (`add_tick_features`)

The system computes advanced microstructure features:

| Feature Category | Examples |
|------------------|----------|
| Price dynamics | Price change, absolute change, log returns |
| Rolling activity | Rolling avg price change, volume, buy/sell ratio |
| Trader behavioral metrics | Trader activity score (inverse time gap between trades) |

All resulting features are normalized with `StandardScaler`.

---

## 6. Anomaly Detection Models

### 6.1 Isolation Forest (`IFDetector`)
- Detects point anomalies (e.g., sudden price/volume shocks)
- Outputs: `IF_Score ∈ [0,1]`  
  Higher score → more anomalous tick

### 6.2 Autoencoder or PCA (`AutoencoderDetector`)
- If PyTorch installed → Autoencoder is trained
- Otherwise → PCA fallback
- Outputs: `AE_Score ∈ [0,1]`  
  Higher score → behavior deviates from norm (algorithmic trading pattern)

---

## 7. Graph-Based Trader Interaction Analysis

### 7.1 Interaction Graph (`build_interaction_graph`)
- Nodes → Traders
- Edges → Trades happening:
  - In proximity (within 15 seconds by default)
  - With correlated direction or similar volumes

### 7.2 Spectral Clustering (`spectral_cluster_traders`)
Traders are grouped into 3 clusters:
1. **Collusive Cluster**
2. **Potentially Suspicious**
3. **Normal Cluster**

Cluster assignment is based on average anomaly metrics + network structure.

---

## 8. Sliding Window Analysis & Labeling

PAMMS breaks the tick stream into overlapping time windows and summarizes:

| Metric | Description |
|--------|---------|
| `IF_mean`, `AE_mean` | Average anomaly level within the window |
| `Top Traders` | Most suspicious IDs based on IF & AE |
| `side_imbalance` | Ratio of Buys vs Sells |

Windows are labeled based on anomaly thresholds:

| Assigned Label | Description |
|----------------|---------|
| `Pump&Exit (Collusive)` | Collusive cluster + high anomaly activity |
| `Spoofing Burst` | High anomaly + suspicious cluster |
| `Algorithmic Anomaly` | Both IF + AE high, no collusion |
| `Volume Shock / Point Anomaly` | IF high only |
| `Temporal Pattern` | AE high only |
| `Normal` | No anomaly |

---

## 9. Output Artifacts

Generated inside `pamms_outputs/`:

### CSV Files
| File | Content |
|------|---------|
| `input_ticks.csv` | Cleaned + normalized raw ticks (Excel snapshot) |
| `synthetic_ticks_scored.csv` | *(name preserved)* scored ticks with IF/AE/cluster labels |
| `windows_summary.csv` | Sliding window results + assigned anomaly label |
| `top_events.csv` | Top anomalous tick events |

### Charts
| File | Description |
|------|-------------|
| `if_scores.png` | Isolation Forest anomaly over time |
| `ae_scores.png` | Autoencoder/PCA anomaly over time |
| `price_with_flags.png` | Price chart with anomalies marked |
| `trader_graph.png` | Network of suspicious trader interactions |

---

## 10. Main Pipeline (`main`)

Execution flow:
1. Load real Excel file
2. Normalize schema → standard columns
3. Feature engineering
4. Isolation Forest + Autoencoder/PCA scoring
5. Graph construction and clustering
6. Sliding window detection + labeling
7. Export CSVs and visual outputs



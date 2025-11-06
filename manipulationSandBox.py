import os
import re
import math
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering

import networkx as nx

USE_TORCH = True
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    USE_TORCH = True
except Exception:
    USE_TORCH = False

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:  
    SCRIPT_DIR = Path.cwd()
OUTDIR = SCRIPT_DIR / "pamms_outputs"
OUTDIR.mkdir(parents=True, exist_ok=True)

# =========================
# Params
# =========================
TH_IF = 0.60         # IF Score >= risk
TH_AE = 0.80         # AE Score >= algorithmic
TOP_K_EVENTS = 99999999   
WINDOW_SEC = 60      
STEP_SEC = 25        # Overlap
MIN_EVENTS = 900
TOP_K_PLOT = 15
FIGSIZE = (9, 6)
DPI_SAVE = 200

# =========================
# Helpers: Schema Normalization
# =========================
def _strip_lower(s):
    import re as _re
    return _re.sub(r"\s+", "", str(s)).lower()

def normalize_tick_schema(df: pd.DataFrame) -> pd.DataFrame:
#    Time, Price, Volume, Side, TraderID, Symbol
    if df is None or df.empty:
        raise ValueError("DataFrame is empty; nothing to normalize.")

    CANDS = {
        "time": ["time","timestamp","datetime","date","تاریخ","زمان","تاریخ-زمان","date_time","dt","ساعت","DateTime"],
        "price": ["price","last","close","قیمت","قيمت","آخرین","اخرین","آخرینقیمت","قیمت پایانی","پایانی"],
        "volume": ["volume","vol","qty","quantity","حجم","تعداد"],
        "side": ["side","bs","direction","type","op","operation","خرید/فروش","جهت","سمت","buy/sell","OrderType"],
        "trader": ["traderid","trader","account","acct","client","clientid","کد","شناسه","شناسه‌معاملاتی","شناسه_معاملاتی","TraderID"],
        "symbol": ["symbol","ticker","نماد","سیمبل","symbolid","symbol_id"],
    }

    cols_map = {c: _strip_lower(c) for c in df.columns}
    inv_map = {v: k for k, v in cols_map.items()}  # normalized -> original

    def find_col(target_key):
        # exact match
        for norm, orig in inv_map.items():
            for cand in CANDS[target_key]:
                if norm == _strip_lower(cand):
                    return orig
        # contains match
        for norm, orig in inv_map.items():
            for cand in CANDS[target_key]:
                if _strip_lower(cand) in norm:
                    return orig
        return None

    # Time
    col_time = find_col("time")
    if col_time is None:
        # date + time separate?
        date_col = None
        time_col = None
        for norm, orig in inv_map.items():
            if any(w in norm for w in ["date","تاریخ"]):
                date_col = orig
            if any(w in norm for w in ["time","زمان","ساعت"]):
                time_col = orig
        if date_col is not None and time_col is not None:
            df["__TMP_TIME__"] = pd.to_datetime(
                df[date_col].astype(str) + " " + df[time_col].astype(str), errors="coerce"
            )
            col_time = "__TMP_TIME__"
        else:
            # pick first datetime-like column
            dt_like = [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)]
            if dt_like:
                col_time = dt_like[0]
    if col_time is None:
        raise KeyError("cannot find date column")

    # Price
    col_price = find_col("price")
    if col_price is None:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        col_price = numeric_cols[0] if numeric_cols else None
    if col_price is None:
        raise KeyError("cannot find price column")

    # Volume
    col_volume = find_col("volume")
    if col_volume is None:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != col_price]
        col_volume = numeric_cols[0] if numeric_cols else None
    if col_volume is None:
        raise KeyError("cannot find Volume column")

    # Side
    col_side = find_col("side")
    if col_side is None:
        df["__TMP_SIDE__"] = "B"
        col_side = "__TMP_SIDE__"

    # TraderID
    col_trader = find_col("trader")
    if col_trader is None:
        df["__TMP_TRADER__"] = [f"T{(i%20)+1:03d}" for i in range(len(df))]
        col_trader = "__TMP_TRADER__"

    # Symbol
    col_symbol = find_col("symbol")
    if col_symbol is None:
        df["__TMP_SYMBOL__"] = "SYNB"
        col_symbol = "__TMP_SYMBOL__"

    out = pd.DataFrame()
    out["Time"] = pd.to_datetime(df[col_time], errors="coerce")
    out["Price"] = pd.to_numeric(df[col_price], errors="coerce")
    out["Volume"] = pd.to_numeric(df[col_volume], errors="coerce")
    out["Side"] = df[col_side].astype(str)

    def map_side(x):
        x = str(x).strip().lower()
        if x in ["b","buy","خرید","kharid","1","long"]:
            return "B"
        if x in ["s","sell","فروش","forush","-1","short"]:
            return "S"
        try:
            v = float(x)
            return "B" if v >= 0 else "S"
        except Exception:
            return "B"
    out["Side"] = out["Side"].map(map_side)

    out["TraderID"] = df[col_trader].astype(str)
    out["Symbol"] = df[col_symbol].astype(str)

    out = out.dropna(subset=["Time","Price","Volume"]).reset_index(drop=True)
    return out

# -------------------------------
# Data Ingestor 
# -------------------------------
class DataIngestor:
    def __init__(self, local_path: str = None):
        if local_path:
            self.local_path = local_path
        else:
            self.local_path = str(SCRIPT_DIR / "sampleData.xlsx")

    def load_local_excel(self):
        p = Path(self.local_path)
        if not p.exists():
            raise FileNotFoundError(
                f"Excel not found at: {p.resolve()}"
            )
        try:
            try:
                import openpyxl  # noqa
                df = pd.read_excel(p, engine="openpyxl")
            except Exception:
                df = pd.read_excel(p)
            if df is None or df.empty:
                raise ValueError("empty file!")
            return df
        except Exception as e:
            raise RuntimeError(f"Error reading Excel: {e}")

    def fetch_tsetmc(self, symbol: str, start_date: str, end_date: str):
        return None

# -------------------------------
# Synthetic Tick Data Generator
# -------------------------------
def generate_synthetic_ticks(n_normal=400, n_spoof=300, n_pnd=100, seed=42):
    rng = np.random.default_rng(seed)
    start = datetime(2024, 1, 1, 9, 0, 0)
    records = []
    symbol = "SYNB"
    price = 100.0
    time_cursor = start
    def tid(i): return f"T{i:03d}"
    normal_traders = [tid(i) for i in range(1, 16)]
    for _ in range(n_normal):
        time_cursor += timedelta(seconds=int(rng.integers(1, 8)))
        price_change = rng.normal(0, 0.1)
        price = max(10, price + price_change)
        vol = int(max(1, rng.lognormal(mean=3, sigma=0.5)))
        side = "B" if rng.random() < 0.5 else "S"
        trader = rng.choice(normal_traders)
        records.append({"Time": time_cursor,"Price": round(price, 2),"Volume": vol,"Side": side,"TraderID": trader,"Symbol": symbol})
    sp_traders = [tid(201), tid(202)]
    for k in range(3):
        base_side = "B" if k % 2 == 0 else "S"
        for _ in range(n_spoof // 3):
            time_cursor += timedelta(seconds=int(rng.integers(1, 3)))
            vol = int(rng.integers(5000, 12000))
            price += 0.2 if base_side == "B" else -0.2
            records.append({"Time": time_cursor,"Price": round(price, 2),"Volume": vol,"Side": base_side,"TraderID": sp_traders[k % 2],"Symbol": symbol})
        for _ in range(2):
            time_cursor += timedelta(seconds=1)
            price += -0.15 if base_side == "B" else 0.15
            records.append({"Time": time_cursor,"Price": round(price, 2),"Volume": int(rng.integers(200, 600)),"Side": "S" if base_side == "B" else "B","TraderID": sp_traders[k % 2],"Symbol": symbol})
    ring = [tid(301), tid(302), tid(303)]
    for j in range(n_pnd // 2):
        time_cursor += timedelta(seconds=int(rng.integers(1, 4)))
        price += abs(rng.normal(0.25, 0.07))
        vol = int(rng.integers(2000, 5000))
        trader = ring[j % len(ring)]
        records.append({"Time": time_cursor,"Price": round(price, 2),"Volume": vol,"Side": "B","TraderID": trader,"Symbol": symbol})
    for j in range(n_pnd // 2):
        time_cursor += timedelta(seconds=int(rng.integers(1, 4)))
        price -= abs(rng.normal(0.35, 0.09))
        price = max(10, price)
        vol = int(rng.integers(2500, 6000))
        trader = ring[j % len(ring)]
        records.append({"Time": time_cursor,"Price": round(price, 2),"Volume": vol,"Side": "S","TraderID": trader,"Symbol": symbol})
    df = pd.DataFrame.from_records(records).sort_values("Time").reset_index(drop=True)
    return df

# -------------------------------
# Feature Engineering
# -------------------------------
def add_tick_features(df: pd.DataFrame):
    df = df.copy()
    df["Time"] = pd.to_datetime(df["Time"])
    df["Epoch"] = df["Time"].astype("int64") // 10**9
    df["Price_Change"] = df["Price"].diff().fillna(0)
    df["Abs_Price_Change"] = df["Price_Change"].abs()
    df["LogRet"] = np.log(df["Price"].clip(lower=1.0)).diff().fillna(0)
    df["Side"] = df["Side"].astype(str).str.upper()
    df["Side_Buy"] = (df["Side"] == "B").astype(int)

    for win in [3, 5, 10]:
        df[f"RollVol_{win}"] = df["Volume"].rolling(win).mean().bfill()
        df[f"RollAbsP_{win}"] = df["Abs_Price_Change"].rolling(win).mean().bfill()
        df[f"RollBuyRatio_{win}"] = df["Side_Buy"].rolling(win).mean().bfill()

    df["TraderID"] = df["TraderID"].astype(str)
    df["Trader_Activity"] = (
        df.groupby("TraderID")["Epoch"].diff().fillna(df["Epoch"].diff().median()).clip(lower=1)
    )
    df["Trader_Activity"] = 1.0 / df["Trader_Activity"]

    feature_cols = [
        "Price","Volume","Price_Change","Abs_Price_Change","LogRet",
        "RollVol_3","RollVol_5","RollVol_10",
        "RollAbsP_3","RollAbsP_5","RollAbsP_10",
        "RollBuyRatio_3","RollBuyRatio_5","RollBuyRatio_10",
        "Trader_Activity"
    ]

    df_features = df[feature_cols].astype("float64").copy()
    scaler = StandardScaler()
    df_features.loc[:, :] = scaler.fit_transform(df_features.to_numpy())

    return df, df_features, scaler

# -------------------------------
# Isolation Forest Detector
# -------------------------------
class IFDetector:
    def __init__(self, contamination=0.12, random_state=42):
        self.model = IsolationForest(
            n_estimators=300,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )
    def fit_score(self, X: pd.DataFrame):
        self.model.fit(X)
        raw = -self.model.decision_function(X)  # larger = more anomalous
        mn, mx = raw.min(), raw.max()
        score = (raw - mn) / (mx - mn + 1e-9)
        return score

# -------------------------------
# Autoencoder (PyTorch or PCA fallback)
# -------------------------------
class AutoencoderDetector:
    def __init__(self, input_dim, bottleneck=8, epochs=30, lr=1e-3, seed=42):
        self.input_dim = input_dim
        self.bottleneck = bottleneck
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.is_torch = USE_TORCH
        if self.is_torch:
            torch.manual_seed(seed)
            self._build_torch()
        else:
            self.pca = PCA(n_components=min(bottleneck, input_dim))

    def _build_torch(self):
        class AE(nn.Module):
            def __init__(self, d, b):
                super().__init__()
                self.enc = nn.Sequential(
                    nn.Linear(d, max(16, b*2)),
                    nn.ReLU(),
                    nn.Linear(max(16, b*2), b)
                )
                self.dec = nn.Sequential(
                    nn.Linear(b, max(16, b*2)),
                    nn.ReLU(),
                    nn.Linear(max(16, b*2), d)
                )
            def forward(self, x):
                z = self.enc(x)
                out = self.dec(z)
                return out
        self.model = AE(self.input_dim, self.bottleneck)
        self.opt = optim.Adam(self.model.parameters(), lr=self.lr)
        self.crit = nn.MSELoss()

    def fit(self, X: np.ndarray):
        if self.is_torch:
            self.model.train()
            Xtensor = torch.tensor(X, dtype=torch.float32)
            for _ in range(self.epochs):
                self.opt.zero_grad()
                out = self.model(Xtensor)
                loss = self.crit(out, Xtensor)
                loss.backward()
                self.opt.step()
        else:
            self.pca.fit(X)

    def score(self, X: np.ndarray):
        if self.is_torch:
            self.model.eval()
            with torch.no_grad():
                Xtensor = torch.tensor(X, dtype=torch.float32)
                out = self.model(Xtensor)
                err = ((out - Xtensor) ** 2).mean(dim=1).numpy()
        else:
            X_proj = self.pca.inverse_transform(self.pca.transform(X))
            err = ((X - X_proj) ** 2).mean(axis=1)
        mn, mx = err.min(), err.max()
        score = (err - mn) / (mx - mn + 1e-12)
        return score

# -------------------------------
# Graph-based Collusion
# -------------------------------
def build_interaction_graph(df: pd.DataFrame, time_window_sec=15):
    df = df.sort_values("Time").reset_index(drop=True)
    traders = df["TraderID"].astype(str).unique().tolist()
    G = nx.Graph()
    for t in traders:
        G.add_node(t)

    epochs = df["Time"].astype("int64") // 10**9
    df = df.copy()
    df["_Epoch"] = epochs

    for i in range(len(df)):
        ti = df.loc[i, "_Epoch"]
        wi = str(df.loc[i, "TraderID"])
        si = df.loc[i, "Side"]
        voli = int(df.loc[i, "Volume"])
        j = i + 1
        while j < len(df):
            tj = df.loc[j, "_Epoch"]
            if tj - ti > time_window_sec:
                break
            wj = str(df.loc[j, "TraderID"])
            sj = df.loc[j, "Side"]
            volj = int(df.loc[j, "Volume"])
            coord = (si == sj) or (abs(voli - volj) < 1000 and si != sj)
            if wi != wj and coord:
                if G.has_edge(wi, wj):
                    G[wi][wj]["weight"] += 1
                else:
                    G.add_edge(wi, wj, weight=1)
            j += 1
    return G

def spectral_cluster_traders(G: nx.Graph, n_clusters=3):
    if G.number_of_nodes() == 0:
        return {}
    nodes = list(G.nodes())
    A = nx.to_numpy_array(G, nodelist=nodes, weight="weight")
    if A.sum() == 0:
        return {n: i for i, n in enumerate(nodes)}
    sc = SpectralClustering(n_clusters=n_clusters, affinity="precomputed",
                            random_state=42, assign_labels="kmeans")
    labels = sc.fit_predict(A)
    return {nodes[i]: int(labels[i]) for i in range(len(nodes))}

def tag_suspicious_clusters(df, cluster_map, if_score, ae_score):
    df = df.copy()
    df["IF_Score"] = if_score
    df["AE_Score"] = ae_score
    df["TraderCluster"] = df["TraderID"].map(cluster_map).fillna(-1).astype(int)

    trader_agg = df.groupby("TraderID")[["IF_Score","AE_Score","Volume"]].mean()
    trader_agg["Trades"] = df.groupby("TraderID")["TraderID"].count()
    trader_agg["Cluster"] = trader_agg.index.map(lambda t: cluster_map.get(t, -1))

    cluster_agg = trader_agg.groupby("Cluster")[["IF_Score","AE_Score","Volume","Trades"]].mean()
    cluster_agg = cluster_agg.sort_values(["IF_Score","AE_Score"], ascending=False)

    labels = {}
    if len(cluster_agg) > 0:
        ordered = list(cluster_agg.index)
        for i, c in enumerate(ordered):
            if i == 0:
                labels[c] = "Collusive Cluster"
            elif i == 1:
                labels[c] = "Potentially Suspicious"
            else:
                labels[c] = "Normal Cluster"
    return trader_agg, cluster_agg, labels, df

# -------------------------------
# Orchestrator
# -------------------------------
class PAMMS:
    def run(self, df_ticks: pd.DataFrame, if_contamination=0.12, ae_bottleneck=8):
        required = {"Time","Price","Volume","Side","TraderID","Symbol"}
        missing = required - set(df_ticks.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        dfF, X, scaler = add_tick_features(df_ticks)

        ifd = IFDetector(contamination=if_contamination)
        if_score = ifd.fit_score(X)

        ae = AutoencoderDetector(input_dim=X.shape[1], bottleneck=ae_bottleneck, epochs=30)
        ae.fit(X.values)
        ae_score = ae.score(X.values)

        G = build_interaction_graph(df_ticks, time_window_sec=15)
        cluster_map = spectral_cluster_traders(G, n_clusters=3)
        trader_agg, cluster_agg, labels, df_scored = tag_suspicious_clusters(dfF, cluster_map, if_score, ae_score)

        return {
            "df_scored": df_scored,
            "graph": G,
            "cluster_map": cluster_map,
            "trader_agg": trader_agg,
            "cluster_agg": cluster_agg,
            "cluster_labels": labels
        }

# -------------------------------
# Visualization  
# -------------------------------
def _top_k_points(df_scored, k=TOP_K_PLOT):
    comb = 0.5*df_scored["IF_Score"] + 0.5*df_scored["AE_Score"]
    idx = np.argsort(-comb.values)[:max(1, k)]
    return df_scored.loc[idx].sort_values("Time")

def plot_if_scores(df_scored, path):
    plt.figure(figsize=FIGSIZE)
    plt.plot(df_scored["Time"], df_scored["IF_Score"], linewidth=1)
    top = _top_k_points(df_scored)
    plt.scatter(top["Time"], top["IF_Score"], marker="x", s=25)
    plt.title("Isolation Forest Anomaly Score over Time")
    plt.xlabel("Time"); plt.ylabel("IF Score (0..1)")
    plt.tight_layout()
    plt.savefig(path, dpi=DPI_SAVE); plt.close()

def plot_ae_scores(df_scored, path):
    plt.figure(figsize=FIGSIZE)
    plt.plot(df_scored["Time"], df_scored["AE_Score"], linewidth=1)
    top = _top_k_points(df_scored)
    plt.scatter(top["Time"], top["AE_Score"], marker="x", s=25)
    plt.title("Autoencoder Reconstruction Error over Time")
    plt.xlabel("Time"); plt.ylabel("AE Score (0..1)")
    plt.tight_layout()
    plt.savefig(path, dpi=DPI_SAVE); plt.close()

def plot_price_with_flags(df_scored, path, top_k=TOP_K_PLOT):
    plt.figure(figsize=FIGSIZE)
    plt.plot(df_scored["Time"], df_scored["Price"], linewidth=1.2)
    top = _top_k_points(df_scored, k=top_k)
    plt.scatter(top["Time"], top["Price"], marker="x", s=35)
    plt.title("Price with Top-5 Combined Anomalies Marked")
    plt.xlabel("Time"); plt.ylabel("Price")
    plt.tight_layout()
    plt.savefig(path, dpi=DPI_SAVE); plt.close()

def plot_graph_clusters(G, cluster_map, labels_map, path):
# top 5
    plt.figure(figsize=FIGSIZE)
    if G.number_of_nodes() == 0:
        plt.title("Trader Graph (no edges)")
        plt.tight_layout(); plt.savefig(path, dpi=DPI_SAVE); plt.close(); return

    try:
        df_scored = pd.read_csv(OUTDIR / "synthetic_ticks_scored.csv", parse_dates=["Time"])
    except Exception:
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, node_size=50, width=0.5)
        plt.title("Trader Interaction Graph")
        plt.axis("off"); plt.tight_layout(); plt.savefig(path, dpi=DPI_SAVE); plt.close(); return

    top_points = _top_k_points(df_scored, k=TOP_K_PLOT)
    top_traders = set(top_points["TraderID"].astype(str).tolist())

    nodes_to_keep = [n for n in G.nodes() if str(n) in top_traders]
    H = G.subgraph(nodes_to_keep).copy()

    if H.number_of_nodes() == 0:
        plt.title("Trader Graph (Top-5 events had no shared edges)")
        plt.tight_layout(); plt.savefig(path, dpi=DPI_SAVE); plt.close(); return

    pos = nx.spring_layout(H, seed=42)
    clusters = {}
    for n in H.nodes():
        c = cluster_map.get(n, -1)
        clusters.setdefault(c, []).append(n)

    for c, nodes in clusters.items():
        nx.draw_networkx_nodes(H, pos, nodelist=nodes, node_size=220, alpha=0.9)
    nx.draw_networkx_edges(H, pos, width=1.4, alpha=0.6)

    lbls = {}
    for n in H.nodes():
        c = cluster_map.get(n, -1)
        tag = labels_map.get(c, "Cluster")
        lbls[n] = f"{n}\nC{c}:{tag}"
    nx.draw_networkx_labels(H, pos, labels=lbls, font_size=8)

    plt.title("Trader Interaction Graph (Top-5 Events — Induced Subgraph)")
    plt.axis("off"); plt.tight_layout()
    plt.savefig(path, dpi=DPI_SAVE); plt.close()

# -------------------------------
# Windowed Summary + Labeling
# -------------------------------
def sliding_windows(df: pd.DataFrame, window_sec=30, step_sec=15):
    t = df["Time"].sort_values().to_numpy()
    if len(t) == 0:
        return []
    t0 = t[0]
    tend = t[-1]
    wins = []
    start = t0
    from numpy import timedelta64
    while start <= tend:
        end = start + np.timedelta64(window_sec, 's')
        wins.append((start, end))
        start = start + np.timedelta64(step_sec, 's')
    return wins

def label_window(row, has_collusive, has_spoofer):
    if row["IF_mean"] >= TH_IF and row["AE_mean"] >= TH_AE:
        if has_collusive:
            return "Pump&Exit (Collusive)"
        if has_spoofer:
            return "Spoofing Burst"
        return "Algorithmic Anomaly"
    if row["IF_mean"] >= TH_IF and row["AE_mean"] < TH_AE:
        if has_spoofer:
            return "Spoof-like (Volume Shock)"
        return "Point Anomaly"
    if row["AE_mean"] >= TH_AE and row["IF_mean"] < TH_IF:
        return "Temporal Pattern Anomaly"
    return "Normal"

def summarize_windows(df_scored: pd.DataFrame, cluster_map: dict, window_sec=WINDOW_SEC, step_sec=STEP_SEC):
    wins = sliding_windows(df_scored, window_sec, step_sec)
    summaries = []
    for (ws, we) in wins:
        mask = (df_scored["Time"] >= ws) & (df_scored["Time"] < we)
        chunk = df_scored.loc[mask]
        if chunk.empty:
            continue

        IF_mean = float(chunk["IF_Score"].mean())
        AE_mean = float(chunk["AE_Score"].mean())
        IF_max = float(chunk["IF_Score"].max())
        AE_max = float(chunk["AE_Score"].max())

        buys = (chunk["Side"] == "B").sum()
        sells = (chunk["Side"] == "S").sum()
        side_imbalance = (buys - sells) / max(1, (buys + sells))

        top_traders = (
            chunk.groupby("TraderID")[["IF_Score","AE_Score","Volume"]]
            .mean()
            .sort_values(["IF_Score","AE_Score"], ascending=False)
            .head(5)
        )
        top_trader_list = ", ".join(top_traders.index.tolist())

        traders_in_win = set(chunk["TraderID"].unique().tolist())
        clusters_in_win = {cluster_map.get(t, -1) for t in traders_in_win}
        has_collusive = any(c == 2 for c in clusters_in_win)  # 2: Collusive
        has_spoofer = any(c == 1 for c in clusters_in_win)    # 1: Potentially Suspicious

        lab = label_window(
            {"IF_mean": IF_mean, "AE_mean": AE_mean},
            has_collusive, has_spoofer
        )

        summaries.append({
            "window_start": pd.to_datetime(ws),
            "window_end": pd.to_datetime(we),
            "rows": int(len(chunk)),
            "IF_mean": IF_mean,
            "AE_mean": AE_mean,
            "IF_max": IF_max,
            "AE_max": AE_max,
            "side_imbalance": float(side_imbalance),
            "top_traders": top_trader_list,
            "label": lab
        })
    return pd.DataFrame(summaries).sort_values(["IF_mean","AE_mean"], ascending=False).reset_index(drop=True)

def export_top_events(df_scored: pd.DataFrame, out_csv: Path, top_k=TOP_K_EVENTS):
    comb = df_scored["IF_Score"]*0.5 + df_scored["AE_Score"]*0.5
    idx = np.argsort(-comb.values)[:top_k]
    top = df_scored.loc[idx, ["Time","Price","Volume","Side","TraderID","IF_Score","AE_Score"]].copy()
    def label_point(r):
        if r["IF_Score"] >= TH_IF and r["AE_Score"] >= TH_AE:
            return "Pump/Spoof Candidate"
        if r["IF_Score"] >= TH_IF:
            return "Volume/Price Shock"
        if r["AE_Score"] >= TH_AE:
            return "Algorithmic Pattern"
        return "Low"
    top["label_point"] = top.apply(label_point, axis=1)
    top.sort_values(["IF_Score","AE_Score"], ascending=False).to_csv(out_csv, index=False)
    return top

# -------------------------------
# Main 
# -------------------------------
def main():
    # 1) Load data 
    ingestor = DataIngestor()
    df = ingestor.load_local_excel()

    print("[INFO] Original columns:", list(df.columns))

    # 2) Normalize schema
    df = normalize_tick_schema(df)
    print("[INFO] Normalized columns:", list(df.columns))

    # 3) Ensure dtypes & missing keys
    df["Time"] = pd.to_datetime(df["Time"])
    df["Side"] = df["Side"].astype(str).str.upper()
    if "TraderID" not in df.columns:
        df["TraderID"] = ["T{:03d}".format(i%20+1) for i in range(len(df))]
    if "Symbol" not in df.columns:
        df["Symbol"] = "SYNB"

    # 4) Minimum events guard
    if len(df) < MIN_EVENTS:
        print(f"[WARN] فقط {len(df)} records  (<{MIN_EVENTS}) continue with records")

    # 5) Save input snapshot
    (OUTDIR / "input_ticks.csv").write_text(df.to_csv(index=False))

    # 6) Run pipeline
    pamms = PAMMS()
    results = pamms.run(df, if_contamination=0.12, ae_bottleneck=8)

    df_scored = results["df_scored"]
    G = results["graph"]
    cluster_map = results["cluster_map"]
    cluster_labels = results["cluster_labels"]

    # 7) Save scored ticks 
    df_scored.to_csv(OUTDIR / "synthetic_ticks_scored.csv", index=False)

    # 8) Plots 
    plot_if_scores(df_scored, OUTDIR / "if_scores.png")
    plot_ae_scores(df_scored, OUTDIR / "ae_scores.png")
    plot_price_with_flags(df_scored, OUTDIR / "price_with_flags.png")
    plot_graph_clusters(G, cluster_map, cluster_labels, OUTDIR / "trader_graph.png")

    # 9) Windowed summary + Top events CSVs 
    windows_df = summarize_windows(df_scored, cluster_map, WINDOW_SEC, STEP_SEC)
    windows_df.to_csv(OUTDIR / "windows_summary.csv", index=False)

    top_events = export_top_events(df_scored, OUTDIR / "top_events.csv", TOP_K_EVENTS)

    # 10) Print recap
    print("Done. Outputs saved in:", str(OUTDIR.resolve()))
    print("Cluster labels:", cluster_labels)
    print("Num traders:", len(G.nodes()), " Num edges:", len(G.edges()))
    print(f"Windows summary rows: {len(windows_df)} | Top events: {len(top_events)}")
    print(f"Thresholds -> IF: {TH_IF}, AE: {TH_AE}, Window: {WINDOW_SEC}s step {STEP_SEC}s")

if __name__ == "__main__":
    main()

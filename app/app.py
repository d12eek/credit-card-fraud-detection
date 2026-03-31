"""
app/app.py  —  Credit Card Fraud Detection · Streamlit Dashboard
Run with:  streamlit run app/app.py
"""

import os, sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ── Path setup ────────────────────────────────────────────────────────────────
_THIS = os.path.abspath(__file__)        # .../app/app.py
_APP  = os.path.dirname(_THIS)          # .../app/
ROOT  = os.path.dirname(_APP)           # .../Credit Card Fraud detection/
SRC   = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from predict import predict, load_artifacts

# ── Page config — MUST be the very first st.* call ───────────────────────────
st.set_page_config(
    page_title="FraudShield · Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Pre-load models once (after set_page_config) ─────────────────────────────
@st.cache_resource
def get_artifacts():
    return load_artifacts()

ARTIFACTS = get_artifacts()

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:      #0a0e1a;
    --surface: #111827;
    --border:  #1f2d45;
    --accent:  #00e5ff;
    --danger:  #ff4d6d;
    --warn:    #ffa726;
    --safe:    #00e676;
    --text:    #e2e8f0;
    --muted:   #64748b;
}

html, body, .stApp { background-color: var(--bg) !important; }

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}

h1, h2, h3 { font-family: 'Space Mono', monospace !important; color: var(--accent) !important; }
p, label, div { font-family: 'DM Sans', sans-serif !important; color: var(--text) !important; }

[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: 0.08em; }
[data-testid="stMetricValue"] { color: var(--accent) !important; font-family: 'Space Mono', monospace !important; font-size: 1.6rem !important; }

.stButton > button {
    background: linear-gradient(135deg, #00e5ff22, #00e5ff44) !important;
    border: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    font-family: 'Space Mono', monospace !important;
    border-radius: 8px !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #00e5ff44, #00e5ff66) !important;
    box-shadow: 0 0 20px #00e5ff44 !important;
}

.stTextInput input, .stNumberInput input, .stSelectbox select {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: var(--surface);
    border-radius: 10px;
    border: 1px solid var(--border);
    gap: 4px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    border-radius: 8px !important;
    padding: 8px 20px !important;
}
.stTabs [aria-selected="true"] {
    background: var(--accent) !important;
    color: #0a0e1a !important;
}

[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 10px; }

.alert-high   { background:#ff4d6d18; border:1px solid #ff4d6d; border-radius:10px; padding:1rem; }
.alert-medium { background:#ffa72618; border:1px solid #ffa726; border-radius:10px; padding:1rem; }
.alert-low    { background:#ffeb3b18; border:1px solid #ffeb3b; border-radius:10px; padding:1rem; }
.alert-safe   { background:#00e67618; border:1px solid #00e676; border-radius:10px; padding:1rem; }

.verdict-badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 999px;
    font-family: 'Space Mono', monospace;
    font-size: 0.9rem;
    font-weight: bold;
    letter-spacing: 0.05em;
}
.badge-fraud  { background:#ff4d6d33; border:1px solid #ff4d6d; color:#ff4d6d; }
.badge-normal { background:#00e67633; border:1px solid #00e676; color:#00e676; }

hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)


# ── Constants ─────────────────────────────────────────────────────────────────
ALERT_CONFIG = {
    "🔴 HIGH":   ("alert-high",   "#ff4d6d", "FRAUD DETECTED"),
    "🟠 MEDIUM": ("alert-medium", "#ffa726", "SUSPICIOUS"),
    "🟡 LOW":    ("alert-low",    "#ffeb3b", "BORDERLINE"),
    "🟢 SAFE":   ("alert-safe",   "#00e676", "NORMAL"),
}

V_COLS   = [f"V{i}" for i in range(1, 29)]

# ── IMPORTANT: column order must match training data exactly ──────────────────
# training order: Time, V1, V2, ..., V28, Amount
TRAIN_COL_ORDER = ["Time"] + V_COLS + ["Amount"]

# For batch upload (also accept Class column which we drop)
ALL_COLS = ["Time", "Amount"] + V_COLS

FRAUD_SAMPLE = {
    "Time": 406.0, "Amount": 2125.87,
    "V1": -3.04, "V2": -3.16, "V3": -1.34, "V4": -0.78, "V5":  0.46,
    "V6": -0.68, "V7": -3.54, "V8":  0.01, "V9": -1.02, "V10":-4.23,
    "V11": 2.02, "V12":-4.99, "V13":-0.65, "V14":-5.68, "V15": 0.52,
    "V16":-2.81, "V17":-9.46, "V18":-2.60, "V19": 0.75, "V20": 0.61,
    "V21": 0.54, "V22": 0.25, "V23":-0.09, "V24": 0.24, "V25": 0.29,
    "V26": 0.41, "V27": 0.31, "V28": 0.14,
}
NORMAL_SAMPLE = {
    "Time": 52000.0, "Amount": 45.20,
    "V1":  1.19, "V2":  0.26, "V3":  0.17, "V4":  0.45, "V5":  0.06,
    "V6": -0.08, "V7":  0.09, "V8": -0.06, "V9": -0.26, "V10":-0.17,
    "V11":-0.14, "V12": 0.07, "V13": 0.03, "V14": 0.22, "V15": 0.07,
    "V16":-0.07, "V17":-0.03, "V18":-0.05, "V19": 0.02, "V20":-0.05,
    "V21":-0.04, "V22": 0.05, "V23": 0.01, "V24": 0.01, "V25":-0.10,
    "V26":-0.02, "V27":-0.03, "V28":-0.01,
}


# ── Chart helpers ─────────────────────────────────────────────────────────────
def gauge_chart(confidence: float, alert: str) -> go.Figure:
    color_map = {
        "🔴 HIGH": "#ff4d6d", "🟠 MEDIUM": "#ffa726",
        "🟡 LOW":  "#ffeb3b", "🟢 SAFE":   "#00e676",
    }
    color = color_map.get(alert, "#00e5ff")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(confidence * 100, 1),
        number={"suffix": "%", "font": {"color": color, "size": 36, "family": "Space Mono"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#64748b", "tickfont": {"color": "#64748b"}},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "#111827",
            "bordercolor": "#1f2d45",
            "steps": [
                {"range": [0,  40], "color": "rgba(0, 230, 118, 0.13)"},
                {"range": [40, 60], "color": "rgba(255, 235, 59, 0.13)"},
                {"range": [60, 80], "color": "rgba(255, 167, 38, 0.13)"},
                {"range": [80,100], "color": "rgba(255, 77, 109, 0.13)"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "thickness": 0.75, "value": confidence * 100},
        },
        title={"text": "Fraud Confidence", "font": {"color": "#64748b", "size": 13, "family": "DM Sans"}},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0", height=260,
        margin=dict(t=40, b=10, l=20, r=20),
    )
    return fig


def bar_chart(scores: dict) -> go.Figure:
    models = list(scores.keys())
    values = [round(v * 100, 1) for v in scores.values()]
    colors = ["#ff4d6d" if v >= 55 else "#ffa726" if v >= 40 else "#00e676" for v in values]
    fig = go.Figure(go.Bar(
        x=models, y=values,
        marker_color=colors,
        marker_line_color="#1f2d45", marker_line_width=1,
        text=[f"{v}%" for v in values],
        textposition="outside",
        textfont={"family": "Space Mono", "color": "#e2e8f0", "size": 12},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0", height=240,
        yaxis=dict(range=[0, 110], showgrid=True, gridcolor="#1f2d45", color="#64748b"),
        xaxis=dict(color="#64748b"),
        margin=dict(t=20, b=10, l=10, r=10),
        showlegend=False,
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ FraudShield")
    st.markdown("<p style='color:#64748b;font-size:0.8rem'>Credit Card Anomaly Detection</p>",
                unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Models Active")
    st.markdown("✅ &nbsp; Isolation Forest", unsafe_allow_html=True)
    st.markdown("✅ &nbsp; One-Class SVM",    unsafe_allow_html=True)
    st.markdown("✅ &nbsp; LOF",              unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Ensemble Weights")
    st.markdown("""
    <small style='color:#64748b'>
    • Isolation Forest : 50%<br>
    • LOF             : 30%<br>
    • One-Class SVM   : 20%
    </small>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        "<small style='color:#64748b'>Dataset: 284,807 transactions<br>Fraud rate: 0.173%</small>",
        unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🛡️ FraudShield")
st.markdown(
    "<p style='color:#64748b;margin-top:-10px'>"
    "Real-time credit card fraud detection · Ensemble anomaly detection</p>",
    unsafe_allow_html=True)
st.markdown("---")

tab1, tab2, tab3 = st.tabs([
    "  🔍 Single Transaction  ",
    "  📁 Batch Upload  ",
    "  📊 Model Performance  ",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single Transaction
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Enter Transaction Details")
    st.markdown(
        "<p style='color:#64748b;font-size:0.85rem'>"
        "Fill in the transaction features below. V1–V28 are PCA-transformed features.</p>",
        unsafe_allow_html=True)
    st.markdown("")

    # Preset buttons
    col_p1, col_p2, _ = st.columns([1, 1, 3])
    with col_p1:
        if st.button("📋 Load Normal Sample"):
            st.session_state["preset"] = "normal"
    with col_p2:
        if st.button("🚨 Load Fraud Sample"):
            st.session_state["preset"] = "fraud"

    preset   = st.session_state.get("preset", None)
    defaults = FRAUD_SAMPLE if preset == "fraud" else NORMAL_SAMPLE if preset == "normal" else {}

    # Input form
    st.markdown("")
    with st.expander("⚙️ Basic Fields", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            time_val = st.number_input(
                "Time (seconds since first transaction)",
                value=float(defaults.get("Time", 50000.0)), step=1.0)
        with c2:
            amount_val = st.number_input(
                "Amount (€)",
                value=float(defaults.get("Amount", 45.0)),
                step=0.01, format="%.2f")

    v_vals = {}
    with st.expander("🔢 V1 – V14 (PCA Features)", expanded=False):
        cols = st.columns(4)
        for i, v in enumerate([f"V{j}" for j in range(1, 15)]):
            with cols[i % 4]:
                v_vals[v] = st.number_input(
                    v, value=float(defaults.get(v, 0.0)),
                    step=0.01, format="%.4f", key=f"inp_{v}")

    with st.expander("🔢 V15 – V28 (PCA Features)", expanded=False):
        cols2 = st.columns(4)
        for i, v in enumerate([f"V{j}" for j in range(15, 29)]):
            with cols2[i % 4]:
                v_vals[v] = st.number_input(
                    v, value=float(defaults.get(v, 0.0)),
                    step=0.01, format="%.4f", key=f"inp_{v}")

    st.markdown("")
    _, btn_col, _ = st.columns([2, 1, 2])
    with btn_col:
        run_btn = st.button("🔍 ANALYSE TRANSACTION", use_container_width=True)

    # Result
    if run_btn:
        # ── FIX: build DataFrame in TRAINING column order: Time, V1–V28, Amount ──
        ordered_row = {"Time": time_val}
        ordered_row.update(v_vals)               # V1 through V28
        ordered_row["Amount"] = amount_val
        df_input = pd.DataFrame([ordered_row])[TRAIN_COL_ORDER]

        with st.spinner("Running ensemble models..."):
            results = predict(df_input, ARTIFACTS)

        r          = results.iloc[0]
        label      = r["label"]
        confidence = r["confidence"]
        alert      = r["alert"]

        scores_dict = {
            "Isolation Forest": float(confidence),
            "One-Class SVM":    float(confidence),
            "LOF":              float(confidence),
        }

        st.markdown("---")
        st.markdown("### 🧾 Verdict")

        res_c1, res_c2, res_c3 = st.columns(3)

        with res_c1:
            cfg       = ALERT_CONFIG.get(alert, ("alert-safe", "#00e676", "NORMAL"))
            badge_cls = "badge-fraud" if label == "FRAUD" else "badge-normal"
            st.markdown(f"""
            <div class='{cfg[0]}'>
                <p style='margin:0;font-size:0.75rem;color:#64748b;
                          text-transform:uppercase;letter-spacing:0.1em'>VERDICT</p>
                <p style='margin:4px 0 0 0;font-size:1.6rem;font-family:Space Mono;
                          color:{cfg[1]};font-weight:bold'>{cfg[2]}</p>
                <span class='verdict-badge {badge_cls}' style='margin-top:8px'>{label}</span>
                <p style='margin:8px 0 0 0;font-size:0.8rem;color:#64748b'>Alert level: {alert}</p>
            </div>
            """, unsafe_allow_html=True)

        with res_c2:
            st.plotly_chart(gauge_chart(confidence, alert),
                            use_container_width=True, config={"displayModeBar": False})

        with res_c3:
            st.markdown(
                "<p style='color:#64748b;font-size:0.75rem;text-transform:uppercase;"
                "letter-spacing:0.1em;margin-bottom:8px'>Ensemble confidence</p>",
                unsafe_allow_html=True)
            st.plotly_chart(bar_chart(scores_dict),
                            use_container_width=True, config={"displayModeBar": False})

        st.markdown("##### 📌 Key Features Submitted")
        key_df = pd.DataFrame({
            "Feature": ["Time", "Amount", "V14 (top fraud signal)",
                        "V17 (top fraud signal)", "V12"],
            "Value":   [time_val, amount_val,
                        v_vals.get("V14", 0.0),
                        v_vals.get("V17", 0.0),
                        v_vals.get("V12", 0.0)],
        })
        st.dataframe(key_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Batch Upload
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Batch Transaction Screening")
    st.markdown(
        "<p style='color:#64748b;font-size:0.85rem'>"
        "Upload a CSV with columns: Time, Amount, V1–V28. The Class column is optional.</p>",
        unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    if uploaded:
        df_up = pd.read_csv(uploaded)
        st.markdown(f"**Loaded:** {len(df_up):,} rows · {df_up.shape[1]} columns")

        missing = [c for c in ALL_COLS if c not in df_up.columns]
        if missing:
            st.error(f"❌ Missing columns: {missing}")
        else:
            with st.spinner(f"Scoring {len(df_up):,} transactions..."):
                # ── FIX: reorder columns to match training order ──────────────
                df_feat = df_up[TRAIN_COL_ORDER].copy()
                results = predict(df_feat, ARTIFACTS)

            df_out = pd.concat(
                [df_up.reset_index(drop=True),
                 results[["label", "confidence", "alert"]]],
                axis=1)

            n_fraud = (results["label"] == "FRAUD").sum()
            n_total = len(results)
            pct     = n_fraud / n_total * 100

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Transactions", f"{n_total:,}")
            m2.metric("Flagged as Fraud",   f"{n_fraud:,}")
            m3.metric("Fraud Rate",          f"{pct:.2f}%")
            m4.metric("High Alerts",         f"{(results['alert'] == '🔴 HIGH').sum():,}")

            st.markdown("---")

            alert_counts = results["alert"].value_counts().reset_index()
            alert_counts.columns = ["Alert", "Count"]
            cmap_pie = {
                "🔴 HIGH": "#ff4d6d", "🟠 MEDIUM": "#ffa726",
                "🟡 LOW":  "#ffeb3b", "🟢 SAFE":   "#00e676",
            }
            fig_pie = px.pie(alert_counts, values="Count", names="Alert",
                             color="Alert", color_discrete_map=cmap_pie, hole=0.55)
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0", legend=dict(font=dict(color="#e2e8f0")),
                height=300, margin=dict(t=10, b=10),
            )

            col_pie, col_table = st.columns([1, 2])
            with col_pie:
                st.markdown("**Alert Distribution**")
                st.plotly_chart(fig_pie, use_container_width=True,
                                config={"displayModeBar": False})
            with col_table:
                st.markdown("**Flagged Transactions (top 50)**")
                flagged = df_out[df_out["label"] == "FRAUD"].head(50)
                st.dataframe(
                    flagged[["Time", "Amount", "label", "confidence", "alert"]],
                    use_container_width=True, hide_index=True)

            csv_out = df_out.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Full Results CSV",
                               data=csv_out, file_name="fraud_results.csv",
                               mime="text/csv")
    else:
        st.info("👆 Upload a CSV file to begin batch scoring. Use the creditcard.csv format.")
        st.markdown("**Expected column format:**")
        st.dataframe(pd.DataFrame(columns=ALL_COLS), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Performance
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Model Performance Summary")
    st.markdown(
        "<p style='color:#64748b;font-size:0.85rem'>"
        "Scores from evaluation on 85,443 held-out transactions (148 frauds).</p>",
        unsafe_allow_html=True)

    perf_data = {
        "Model":        ["Isolation Forest", "One-Class SVM", "LOF"],
        "ROC-AUC":      [0.9522, 0.8056, 0.9487],
        "PR-AUC":       [0.1508, 0.1975, 0.0643],
        "F2-Score":     [0.3747, 0.4256, 0.2605],
        "Fraud Caught": [90, 79, 82],
        "Fraud Missed": [58, 69, 66],
        "False Alarms": [519, 257, 900],
        "Precision":    [0.15, 0.24, 0.08],
        "Recall":       [0.61, 0.53, 0.55],
    }
    df_perf = pd.DataFrame(perf_data)

    st.dataframe(
        df_perf.style
               .background_gradient(subset=["ROC-AUC", "PR-AUC", "F2-Score"], cmap="Blues")
               .format({"ROC-AUC": "{:.4f}", "PR-AUC": "{:.4f}", "F2-Score": "{:.4f}",
                        "Precision": "{:.2f}", "Recall": "{:.2f}"}),
        use_container_width=True, hide_index=True,
    )

    st.markdown("---")
    ch1, ch2 = st.columns(2)

    with ch1:
        st.markdown("**ROC-AUC by Model**")
        fig_roc = go.Figure(go.Bar(
            x=df_perf["Model"], y=df_perf["ROC-AUC"],
            marker_color=["#00e5ff", "#7c3aed", "#f97316"],
            text=[f"{v:.4f}" for v in df_perf["ROC-AUC"]],
            textposition="outside",
            textfont={"color": "#e2e8f0", "family": "Space Mono"},
        ))
        fig_roc.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(range=[0, 1.05], gridcolor="#1f2d45", color="#64748b"),
            xaxis=dict(color="#64748b"), font_color="#e2e8f0",
            height=280, margin=dict(t=20, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_roc, use_container_width=True, config={"displayModeBar": False})

    with ch2:
        st.markdown("**Fraud Caught vs False Alarms**")
        fig_fb = go.Figure()
        fig_fb.add_trace(go.Bar(name="Fraud Caught", x=df_perf["Model"],
                                y=df_perf["Fraud Caught"], marker_color="#00e676"))
        fig_fb.add_trace(go.Bar(name="False Alarms", x=df_perf["Model"],
                                y=df_perf["False Alarms"], marker_color="#ff4d6d"))
        fig_fb.update_layout(
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(gridcolor="#1f2d45", color="#64748b"),
            xaxis=dict(color="#64748b"), font_color="#e2e8f0",
            legend=dict(font=dict(color="#e2e8f0")),
            height=280, margin=dict(t=20, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_fb, use_container_width=True, config={"displayModeBar": False})

    st.markdown("**Precision vs Recall Trade-off**")
    fig_pr  = go.Figure()
    clrs_pr = ["#00e5ff", "#7c3aed", "#f97316"]
    for i, row in df_perf.iterrows():
        fig_pr.add_trace(go.Scatter(
            x=[row["Precision"]], y=[row["Recall"]],
            mode="markers+text",
            marker=dict(size=20, color=clrs_pr[i], opacity=0.85,
                        line=dict(color="#0a0e1a", width=2)),
            text=[row["Model"]], textposition="top center",
            textfont=dict(color=clrs_pr[i], size=11, family="Space Mono"),
            name=row["Model"],
        ))
    fig_pr.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Precision", range=[0, 0.5], gridcolor="#1f2d45", color="#64748b"),
        yaxis=dict(title="Recall",    range=[0, 0.9], gridcolor="#1f2d45", color="#64748b"),
        font_color="#e2e8f0", legend=dict(font=dict(color="#e2e8f0")),
        height=320, margin=dict(t=20, b=40, l=40, r=20),
    )
    st.plotly_chart(fig_pr, use_container_width=True, config={"displayModeBar": False})

    st.markdown("---")
    st.markdown("""
    <div style='background:#111827;border:1px solid #1f2d45;border-radius:10px;padding:1rem 1.5rem'>
    <p style='color:#64748b;font-size:0.75rem;text-transform:uppercase;
              letter-spacing:0.1em;margin:0 0 8px 0'>Why these scores?</p>
    <p style='font-size:0.85rem;color:#94a3b8;margin:0'>
    These are <b>unsupervised anomaly models</b> — they never see a fraud label during training.
    The high ROC-AUC (0.95) shows the models rank fraud correctly, but low PR-AUC reflects
    the extreme class imbalance (0.17% fraud). One-Class SVM achieves the best precision (0.24)
    while Isolation Forest catches the most frauds (90/148). The ensemble combines all three
    for the best real-world balance.
    </p>
    </div>
    """, unsafe_allow_html=True)
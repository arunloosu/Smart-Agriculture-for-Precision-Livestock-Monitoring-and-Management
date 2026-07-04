"""
Precision Livestock Monitoring — Operations Dashboard.

Run:
    streamlit run app.py
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import data_loader as dl

st.set_page_config(page_title="Precision Livestock Dashboard", page_icon="🐄", layout="wide")


def demo_badge(is_real: bool):
    if not is_real:
        st.markdown(
            "<span style='background-color:#FFF3CD;color:#856404;padding:2px 8px;"
            "border-radius:4px;font-size:0.8em;'>DEMO DATA — run this level's pipeline "
            "to see live results</span>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<span style='background-color:#D4EDDA;color:#155724;padding:2px 8px;"
            "border-radius:4px;font-size:0.8em;'>LIVE DATA</span>",
            unsafe_allow_html=True,
        )


st.title("🐄 Smart Agriculture — Precision Livestock Monitoring")
st.caption("Operations dashboard aggregating ML, DL, NLP, SLM, Gen AI, and Agentic AI outputs")

tabs = st.tabs([
    "📊 Farm Overview", "🔴 Live Telemetry", "🐮 Behavior Monitoring", "🚨 Health Alerts",
    "📝 Advisory Insights (NLP/SLM)", "🎨 Gen AI Augmentation",
    "🏗️ System Architecture", "🤖 Agent Audit Log",
])

# ---------------------------------------------------------------- Overview
with tabs[0]:
    ml_metrics, ml_real = dl.load_ml_metrics()
    alerts_df, alerts_real = dl.load_alerts()
    behavior_df, _ = dl.load_behavior_timeline()
    genai_metrics, genai_real = dl.load_genai_metrics()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Animals Monitored", behavior_df["animal_id"].nunique())
    col2.metric("Open Alerts (48h)", len(alerts_df))
    best_model = max(ml_metrics, key=lambda k: ml_metrics[k]["f1_macro"])
    col3.metric("Best Behavior Model F1", f"{ml_metrics[best_model]['f1_macro']:.2f}", best_model)
    col4.metric("Critical Alerts", int((alerts_df["severity"] == "critical").sum()))

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Alert Severity Breakdown")
        demo_badge(alerts_real)
        sev_counts = alerts_df["severity"].value_counts().reset_index()
        sev_counts.columns = ["severity", "count"]
        fig = px.pie(sev_counts, names="severity", values="count", hole=0.4,
                      color="severity",
                      color_discrete_map={"low": "#8ecae6", "medium": "#ffb703", "high": "#fb8500", "critical": "#d00000"})
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Model Comparison (Level ML)")
        demo_badge(ml_real)
        comp_df = pd.DataFrame(ml_metrics).T.reset_index().rename(columns={"index": "model"})
        fig2 = px.bar(comp_df, x="model", y=["accuracy", "f1_macro", "roc_auc_ovr"], barmode="group")
        st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------- Live Telemetry (real-time)
with tabs[1]:
    is_live, age_sec, total_rows = dl.live_producer_status()

    if not is_live:
        st.warning(
            "No live producer detected. This tab reads from `realtime/telemetry.db`, "
            "which is written continuously by `realtime/producer.py`. Start it with:\n\n"
            "```bash\ncd realtime\npython producer.py\n```\n\nthen come back to this tab — "
            "it refreshes automatically once data starts arriving."
            + (f"\n\n(Last reading seen {age_sec:.0f}s ago — producer may have stopped.)"
               if age_sec is not None else "")
        )
    else:
        st.markdown(
            "<span style='background-color:#D4EDDA;color:#155724;padding:2px 8px;"
            "border-radius:4px;font-size:0.8em;'>🔴 LIVE — streaming from realtime/producer.py</span>",
            unsafe_allow_html=True,
        )

    refresh_seconds = st.slider("Auto-refresh interval (seconds)", 2, 15, 3, key="live_refresh_rate")

    @st.fragment(run_every=refresh_seconds)
    def render_live_telemetry():
        is_live, age_sec, total_rows = dl.live_producer_status()
        latest = dl.load_live_latest()
        history = dl.load_live_history(minutes=15)
        live_alerts = dl.load_live_alerts(limit=20)

        if latest.empty:
            st.info("Waiting for the first reading from `producer.py`...")
            return

        st.caption(f"{total_rows:,} readings ingested so far · last update "
                   f"{age_sec:.0f}s ago · {latest['animal_id'].nunique()} animals reporting")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg Heart Rate", f"{latest['heart_rate'].mean():.0f} bpm")
        c2.metric("Avg Body Temp", f"{latest['body_temp_c'].mean():.1f} °C")
        c3.metric("Animals Anomalous Now", int(latest["is_anomaly"].sum()))
        c4.metric("Live Alerts Raised", len(live_alerts))

        st.divider()
        col_l, col_r = st.columns([3, 2])
        with col_l:
            st.subheader("Vitals — last 15 minutes")
            metric_choice = st.radio("Metric", ["heart_rate", "body_temp_c", "activity_index"],
                                      horizontal=True, key="live_metric_choice")
            fig = px.line(history, x="ts", y=metric_choice, color="animal_id",
                          render_mode="svg")
            fig.update_layout(showlegend=False, height=360,
                              yaxis_title=metric_choice, xaxis_title="time")
            st.plotly_chart(fig, use_container_width=True, key=f"live_chart_{metric_choice}")

        with col_r:
            st.subheader("Herd Location (geofence)")
            map_fig = px.scatter_mapbox(
                latest, lat="lat", lon="lon", color="is_anomaly",
                hover_name="animal_id", hover_data=["heart_rate", "body_temp_c", "behavior"],
                color_continuous_scale=["#2a9d8f", "#d00000"], zoom=15, height=360,
            )
            map_fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(map_fig, use_container_width=True, key="live_map")

        st.subheader("Current Status — All Animals")
        status_df = latest[["animal_id", "behavior", "heart_rate", "body_temp_c",
                             "activity_index", "is_anomaly"]].rename(
            columns={"is_anomaly": "flagged_anomalous"})
        st.dataframe(
            status_df.style.apply(
                lambda row: ["background-color:#f8d7da" if row["flagged_anomalous"] else "" for _ in row],
                axis=1,
            ),
            use_container_width=True, hide_index=True,
        )

        st.subheader("Live Alert Feed")
        if live_alerts.empty:
            st.caption("No anomalies detected yet — alerts appear here the moment the producer "
                       "injects a fever/tachycardia/inactivity episode.")
        else:
            st.dataframe(live_alerts[["ts", "animal_id", "severity", "condition", "message"]],
                        use_container_width=True, hide_index=True)

    render_live_telemetry()

# ---------------------------------------------------------------- Behavior Monitoring
with tabs[2]:
    behavior_df, behavior_real = dl.load_behavior_timeline()
    demo_badge(behavior_real)

    st.subheader("Behavior Distribution (last 24h, all animals)")
    dist = behavior_df["behavior"].value_counts().reset_index()
    dist.columns = ["behavior", "count"]
    st.plotly_chart(px.bar(dist, x="behavior", y="count", color="behavior"), use_container_width=True)

    st.subheader("Per-Animal Timeline")
    selected_animal = st.selectbox("Select animal", sorted(behavior_df["animal_id"].unique()))
    animal_df = behavior_df[behavior_df["animal_id"] == selected_animal].sort_values("timestamp")
    fig = px.scatter(animal_df, x="timestamp", y="behavior", color="confidence",
                      color_continuous_scale="Viridis", title=f"{selected_animal} — behavior over time")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Confusion Matrix (Level ML, best model)")
    st.caption("Run `level_1_ml/train.py` to generate `outputs/confusion_matrix.png` and it will render here.")
    try:
        st.image("../level_1_ml/outputs/confusion_matrix.png", width=400)
    except Exception:
        st.info("No confusion matrix found yet — showing placeholder. Run Level ML's train.py.")

# ---------------------------------------------------------------- Alerts
with tabs[3]:
    alerts_df, alerts_real = dl.load_alerts()
    demo_badge(alerts_real)

    sev_filter = st.multiselect("Filter by severity", ["low", "medium", "high", "critical"],
                                 default=["medium", "high", "critical"])
    filtered = alerts_df[alerts_df["severity"].isin(sev_filter)].sort_values("timestamp", ascending=False)
    st.dataframe(filtered, use_container_width=True, hide_index=True)

    st.subheader("Alerts per Animal")
    per_animal = alerts_df["animal_id"].value_counts().reset_index()
    per_animal.columns = ["animal_id", "alert_count"]
    st.plotly_chart(px.bar(per_animal, x="animal_id", y="alert_count"), use_container_width=True)

# ---------------------------------------------------------------- Advisory (NLP/SLM)
with tabs[4]:
    digest, digest_real = dl.load_nlp_digest()
    keywords, kw_real = dl.load_keywords()
    slm_samples, slm_real = dl.load_slm_samples()

    st.subheader("District Advisory Digest")
    demo_badge(digest_real)
    for district, dates in digest.items():
        with st.expander(f"📍 {district}"):
            for date, summary in dates.items():
                st.markdown(f"**{date}:** {summary}")

    st.subheader("Emerging Terms (Early Outbreak Signal)")
    demo_badge(kw_real)
    st.write(", ".join(f"`{k}`" for k in keywords))

    st.subheader("SLM-Generated Advisory Samples")
    demo_badge(slm_real)
    for s in slm_samples:
        with st.container(border=True):
            st.markdown(f"**Farmer query:** {s['query']}")
            colA, colB = st.columns(2)
            colA.markdown(f"**Reference (expert) answer:**\n\n{s['reference']}")
            colB.markdown(f"**Model-generated advice:**\n\n{s['generated']}")

# ---------------------------------------------------------------- Gen AI
with tabs[5]:
    genai_metrics, genai_real = dl.load_genai_metrics()
    demo_badge(genai_real)

    if "utility" in genai_metrics:
        st.subheader("Downstream Utility: Does Synthetic Data Help?")
        u = genai_metrics["utility"]
        fig = go.Figure(go.Bar(
            x=["Real data only", "Real + Synthetic"],
            y=[u.get("cnn_accuracy_real_only", 0), u.get("cnn_accuracy_real_plus_synthetic", 0)],
            marker_color=["#adb5bd", "#2a9d8f"],
        ))
        fig.update_layout(yaxis_title="CNN accuracy on held-out real test set")
        st.plotly_chart(fig, use_container_width=True)

    if "robustness" in genai_metrics:
        st.subheader("Robustness Under Field Conditions")
        r = genai_metrics["robustness"]
        fig2 = go.Figure(go.Bar(
            x=["Clean images", "Perturbed (before aug.)", "Perturbed (after aug.)"],
            y=[r.get("clean_accuracy", 0), r.get("perturbed_accuracy_before_augmentation", 0),
               r.get("perturbed_accuracy_after_augmentation", 0)],
            marker_color=["#264653", "#e76f51", "#2a9d8f"],
        ))
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Synthetic Image Samples")
    st.caption("Run `level_6_genai/generate_synthetic.py` to populate this gallery from real output images.")

# ---------------------------------------------------------------- Architecture
with tabs[6]:
    st.subheader("System Architecture (Level LLD)")
    try:
        with open("../level_5_lld/LLD.md") as f:
            st.markdown(f.read())
    except FileNotFoundError:
        st.info("level_5_lld/LLD.md not found relative to this app — run streamlit from the dashboard/ folder.")

# ---------------------------------------------------------------- Agent Audit
with tabs[7]:
    audit_df, audit_real = dl.load_agentic_audit()
    demo_badge(audit_real)

    st.subheader("Agent Decision Trace")
    st.dataframe(audit_df, use_container_width=True, hide_index=True)

    if "action" in audit_df.columns:
        st.subheader("Actions Taken")
        counts = audit_df["action"].value_counts().reset_index()
        counts.columns = ["action", "count"]
        st.plotly_chart(px.bar(counts, x="action", y="count"), use_container_width=True)

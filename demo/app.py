"""Streamlit Interactive Co-Evolution & Platform Policy Dashboard.

Usage:
    streamlit run demo/app.py
"""
import streamlit as st
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import network_evolve as ne

st.set_page_config(page_title="NetworkEvolveLab Simulator", layout="wide")

st.title("NetworkEvolveLab: Dynamic Social Topology & LLM Cognition Simulator")
st.markdown("""
*Visualizing ideological echo chamber emergence, topological phase transitions, and counterfactual platform governance.*
""")

# Sidebar controls
st.sidebar.header("Simulation Parameters")
dataset_choice = st.sidebar.selectbox("Empirical Benchmark Network",
                                       ["Twitter 2020 Election", "SNAP Higgs Twitter", "Reddit ConvoKit"])
theta = st.sidebar.slider("Cognitive Tolerance Threshold (θ)", min_value=0.20, max_value=0.90, value=0.40, step=0.05)
block_diff = st.sidebar.slider("Block / Severance Dissonance Threshold", min_value=0.60, max_value=1.80, value=1.20, step=0.10)
div_rec = st.sidebar.slider("Diversified Recommendation Rate (ρ_div)", min_value=0.00, max_value=0.30, value=0.00, step=0.05)
rounds = st.sidebar.slider("Simulation Rounds", min_value=5, max_value=50, value=20, step=5)

if st.sidebar.button("Run Simulation", type="primary"):
    with st.spinner("Executing co-evolutionary dynamics across network..."):
        # 1. Load dataset（统一使用 src/network_evolve/datasets.py 的加载器）
        n = 60
        if dataset_choice == "Twitter 2020 Election":
            g, ideo = ne.load_twitter_election(num_nodes=n, seed=42)
        elif dataset_choice == "SNAP Higgs Twitter":
            g = ne.load_snap_higgs(num_nodes=n, seed=42)
            rng = np.random.default_rng(42)
            ideo = {i: float(rng.choice([-0.7, 0.7]) + rng.normal(0, 0.15)) for i in range(n)}
        else:
            g, topics = ne.load_reddit_convokit(num_nodes=n, seed=42)
            ideo = {i: float(topics[i][0]) for i in range(n)}

        # 2. 以旗舰引擎 CoEvolutionEngine 运行（导入外部拓扑与信念先验）
        engine = ne.CoEvolutionEngine(
            num_nodes=n,
            seed=42,
            base_tolerance=theta,
            bridger_tolerance=min(1.0, theta + 0.3),
            p_form=0.1,
            theta_attract=0.4,
            heterogeneous=True,
            block_diff=block_diff,
            rec_friction=1.0 - div_rec,
            div_rec=div_rec,
            init_graph=g,
            init_beliefs=ideo,
        )

        history_q, history_pi, history_edges, history_k = [], [], [], []
        for r in range(rounds):
            engine.run_round()
            snap = ne.snapshot(engine)
            history_q.append(snap["modularity_louvain"])
            history_pi.append(snap["polarization"])
            history_edges.append(snap["num_edges"])
            history_k.append(snap["num_louvain_comms"])

        # Display Top Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Final Modularity Q", f"{history_q[-1]:.3f}", delta=f"{history_q[-1] - history_q[0]:+.3f}")
        col2.metric("Final Polarization Π", f"{history_pi[-1]:.3f}")
        col3.metric("Remaining Edges", f"{history_edges[-1]}", delta=f"{history_edges[-1] - history_edges[0]}")
        col4.metric("Louvain Communities K", f"{history_k[-1]:.0f}",
                    help="K 社群下模块度上界为 1 − 1/K（K=2 时 0.500，K=3 时 0.667）")

        # Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        r_axis = np.arange(1, rounds + 1)
        ax1.plot(r_axis, history_q, 'o-', color="#1f77b4", label="Newman Modularity Q (Louvain)")
        ax1.plot(r_axis, history_pi, 's--', color="#d62728", label="Polarization Index Π")
        ax1.axhline(0.500, color="gray", linestyle=":", label="Bipolar bound (K=2): Q ≤ 0.50")
        ax1.set_xlabel("Simulation Round")
        ax1.set_ylabel("Metric Value")
        ax1.set_title("Cognition & Topology Co-Evolution")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend()

        ax2.plot(r_axis, history_edges, '^-', color="#2ca02c", label="Active Directed Edges")
        ax2.set_xlabel("Simulation Round")
        ax2.set_ylabel("Edge Count")
        ax2.set_title("Tie Severance & Unfollowing")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend()

        st.pyplot(fig)
        st.caption("注：Q 的上界随 Louvain 社群数 K 变化（1 − 1/K）；本项目的演化在所有参数网格下停留在双极区间（K≈2）。")
else:
    st.info("Adjust the parameters in the left sidebar and click 'Run Simulation' to observe live phase transitions.")

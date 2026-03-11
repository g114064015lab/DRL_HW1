import streamlit as st
from rl_solver import evaluate_policy
import pandas as pd

# Basic page setup
st.set_page_config(page_title="RL Grid Map Solver", layout="wide")

# CSS Styling to make the Grid look like a map
st.markdown("""
<style>
div[data-testid="column"] {
    display: flex;
    justify-content: center;
    align-items: center;
}
div.stButton > button {
    width: 60px;
    height: 60px;
    margin: 0px !important;
    font-size: 16px;
    font-weight: bold;
    border-radius: 4px;
    border: 1px solid #ccc;
}
/* Style overrides based on markdown labels rendered on button text if possible, 
   but Streamlit buttons don't easily accept color injections from label natively without hacks. 
   We will just rely on emojis to indicate state clearly. */
</style>
""", unsafe_allow_html=True)

st.title("Generate $n \\times n$ Square with RL")

# ----------------------------------------
# 1. State Management
# ----------------------------------------
if 'n' not in st.session_state:
    st.session_state.n = 5
if 'start_id' not in st.session_state:
    st.session_state.start_id = None
if 'end_id' not in st.session_state:
    st.session_state.end_id = None
if 'obstacles' not in st.session_state:
    st.session_state.obstacles = set()
if 'phase' not in st.session_state:
    st.session_state.phase = 0  # 0: start, 1: end, 2: obstacles, 3: done
if 'rl_computed' not in st.session_state:
    st.session_state.rl_computed = False
if 'value_matrix' not in st.session_state:
    st.session_state.value_matrix = None
if 'policy_matrix' not in st.session_state:
    st.session_state.policy_matrix = None

# Sidebar Controls
with st.sidebar:
    st.header("Settings")
    new_n = st.number_input("Enter grid size (5-9):", min_value=5, max_value=9, value=st.session_state.n)
    
    if st.button("Generate New Square") or new_n != st.session_state.n:
        st.session_state.n = new_n
        st.session_state.start_id = None
        st.session_state.end_id = None
        st.session_state.obstacles = set()
        st.session_state.phase = 0
        st.session_state.rl_computed = False
        st.rerun()

n = st.session_state.n
max_obs = n - 2

# Status and Instructions
st.subheader(f"{n} x {n} Square Setup")

if st.session_state.phase == 0:
    st.info("1. Click a cell to set the **Start** position (🍏).")
elif st.session_state.phase == 1:
    st.info("2. Click another cell to set the **End** position (🍎).")
elif st.session_state.phase == 2:
    obs_count = len(st.session_state.obstacles)
    st.info(f"3. Click up to **{max_obs} cells** to set Obstacles (⬛). Selected: {obs_count} / {max_obs}")
else:
    st.success("Grid setup complete! Ready to evaluate RL Policy.")

# Status Bar
col_s, col_e, col_o = st.columns(3)
col_s.metric("Start Cell", st.session_state.start_id if st.session_state.start_id else "Not Set")
col_e.metric("End Cell", st.session_state.end_id if st.session_state.end_id else "Not Set")
col_o.metric("Obstacles", f"{len(st.session_state.obstacles)} / {max_obs}")

# ----------------------------------------
# 2. Grid Rendering & Interaction
# ----------------------------------------
st.write("---")

# Callbacks for button clicks
def cell_clicked(cell_id):
    # Ignore clicks on already selected special cells
    if cell_id == st.session_state.start_id or cell_id == st.session_state.end_id or cell_id in st.session_state.obstacles:
        return
        
    if st.session_state.phase == 0:
        st.session_state.start_id = cell_id
        st.session_state.phase = 1
    elif st.session_state.phase == 1:
        st.session_state.end_id = cell_id
        st.session_state.phase = 2
    elif st.session_state.phase == 2:
        if len(st.session_state.obstacles) < max_obs:
            st.session_state.obstacles.add(cell_id)
            if len(st.session_state.obstacles) == max_obs:
                st.session_state.phase = 3
    
# Render Grid
# To keep cells square-ish, we use st.columns with equal width
for row in range(n):
    cols = st.columns(n)
    for col in range(n):
        cell_id = row * n + col + 1
        with cols[col]:
            # Determine label and style indicators
            label = str(cell_id)
            if cell_id == st.session_state.start_id:
                label = "🍏"
            elif cell_id == st.session_state.end_id:
                label = "🍎"
            elif cell_id in st.session_state.obstacles:
                label = "⬛"
            
            # Button
            st.button(label, key=f"btn_{cell_id}", on_click=cell_clicked, args=(cell_id,))

# ----------------------------------------
# 3. RL Solving
# ----------------------------------------
st.write("---")

if st.session_state.phase >= 2: # Can solve even if max obstacles not reached yet
    if st.button("🚀 Solve RL Policy", type="primary"):
        with st.spinner("Running Iterative Policy Evaluation..."):
            result = evaluate_policy(
                n=st.session_state.n,
                start_id=st.session_state.start_id,
                end_id=st.session_state.end_id,
                obstacle_ids=list(st.session_state.obstacles)
            )
            st.session_state.value_matrix = result['value_matrix']
            st.session_state.policy_matrix = result['policy_matrix']
            st.session_state.rl_computed = True

if st.session_state.rl_computed:
    st.header("RL Results")
    
    col_v, col_p = st.columns(2)
    
    with col_v:
        st.subheader("Value Matrix $V(s)$")
        # Format the dataframe cleanly
        df_v = pd.DataFrame(st.session_state.value_matrix)
        # Apply styling to hide indices and highlight obstacles
        styled_v = df_v.style.format(na_rep="OBS", precision=2) \
            .applymap(lambda v: 'background-color: #64748b; color: white' if pd.isna(v) else '')
        st.dataframe(styled_v, use_container_width=True)
        
    with col_p:
        st.subheader("Policy Matrix $\pi(s)$")
        df_p = pd.DataFrame(st.session_state.policy_matrix)
        styled_p = df_p.style.applymap(lambda v: 'background-color: #64748b; color: white' if v == 'OBS' else 
                                               ('background-color: #ef4444; color: white' if v == 'END' else ''))
        st.dataframe(styled_p, use_container_width=True)

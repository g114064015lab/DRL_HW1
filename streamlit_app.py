import streamlit as st
from rl_solver import value_iteration
import pandas as pd

# Basic page setup
st.set_page_config(page_title="RL Grid Map Solver", layout="wide")

# CSS Styling to make the Grid look like a map
st.markdown("""
<style>
/* Center the grid container and restrict its max-width so columns don't stretch */
[data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
    max-width: 600px;
    margin: 0 auto;
}

/* Override column gaps */
[data-testid="column"] {
    width: 60px !important;
    min-width: 60px !important;
    flex: none !important;
    padding: 0 !important;
    display: flex;
    justify-content: center;
    align-items: center;
}

/* Fix grid button sizes and margins but target only the columns, not sidebar/main buttons */
[data-testid="column"] div.stButton > button {
    width: 60px;
    height: 60px;
    margin: 2px !important;
    font-size: 18px;
    font-weight: bold;
    border-radius: 8px;
    border: 1px solid #e2e8f0;
    padding: 0;
    background-color: white;
    color: #333;
    transition: all 0.2s ease-in-out;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

[data-testid="column"] div.stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    border-color: #cbd5e1;
    background-color: #f8fafc;
}

[data-testid="column"] div.stButton > button:active {
    transform: scale(0.95);
}

/* Value & Policy Matrix Table container styling */
[data-testid="stDataFrame"] {
    width: 100%;
    margin: 0 auto;
}

/* Custom HTML Grid for Optimal Path */
.optimal-grid {
    display: grid;
    gap: 2px;
    background-color: #333;
    padding: 2px;
    margin: 20px auto;
    width: fit-content;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

.optimal-cell {
    width: 60px;
    height: 60px;
    background-color: white;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    font-size: 20px;
    font-weight: bold;
    transition: all 0.3s ease;
}

.optimal-cell.start {
    background-color: #86efac;
    font-size: 14px;
}

.optimal-cell.end {
    background-color: #86efac;
    font-size: 14px;
}

.optimal-cell.path {
    background-color: #86efac;
}

.optimal-cell.obstacle {
    background-color: #64748b;
    color: white;
}

.optimal-cell:hover {
    transform: scale(1.05);
    z-index: 10;
    box-shadow: 0 0 10px rgba(0,0,0,0.2);
}
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


# Value Iteration State
if 'vi_computed' not in st.session_state:
    st.session_state.vi_computed = False
if 'vi_value_matrix' not in st.session_state:
    st.session_state.vi_value_matrix = None
if 'vi_policy_matrix' not in st.session_state:
    st.session_state.vi_policy_matrix = None
if 'optimal_path' not in st.session_state:
    st.session_state.optimal_path = []

# Random Policy State
if 'rp_computed' not in st.session_state:
    st.session_state.rp_computed = False
if 'rp_value_matrix' not in st.session_state:
    st.session_state.rp_value_matrix = None
if 'rp_policy_matrix' not in st.session_state:
    st.session_state.rp_policy_matrix = None

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

        st.session_state.vi_computed = False
        st.session_state.rp_computed = False
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
    st.success("Grid setup complete! Ready to Run Value Iteration.")

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
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🎲 Evaluate Random Policy (HW1-2)", use_container_width=True):
            with st.spinner("Evaluating Random Policy..."):
                from rl_solver import evaluate_policy
                result = evaluate_policy(
                    n=st.session_state.n,
                    start_id=st.session_state.start_id,
                    end_id=st.session_state.end_id,
                    obstacle_ids=list(st.session_state.obstacles)
                )
                st.session_state.rp_value_matrix = result['value_matrix']
                st.session_state.rp_policy_matrix = result['policy_matrix']
                st.session_state.rp_computed = True
    with col_btn2:
        if st.button("👑 Run Value Iteration (Optimal Policy)", type="primary", use_container_width=True):
            with st.spinner("Running Value Iteration..."):
                from rl_solver import value_iteration
                result = value_iteration(
                    n=st.session_state.n,
                    start_id=st.session_state.start_id,
                    end_id=st.session_state.end_id,
                    obstacle_ids=list(st.session_state.obstacles)
                )
                st.session_state.vi_value_matrix = result['value_matrix']
                st.session_state.vi_policy_matrix = result['policy_matrix']
                st.session_state.optimal_path = result['optimal_path']
                st.session_state.vi_computed = True

if st.session_state.rp_computed:
    st.write("---")
    st.header("HW1-2 Random Policy Evaluation")
    col_v1, col_p1 = st.columns(2)
    with col_v1:
        st.subheader("Value Matrix $V^\pi(s)$")
        df_v1 = pd.DataFrame(st.session_state.rp_value_matrix)
        def _style_values1(df):
            styles = pd.DataFrame('', index=df.index, columns=df.columns)
            na_mask = df.isna()
            styles = styles.mask(na_mask, 'background-color: #64748b; color: white')
            return styles
        styled_v1 = df_v1.style.apply(_style_values1, axis=None).format(na_rep="OBS", precision=2)
        st.dataframe(styled_v1, use_container_width=True)
        
    with col_p1:
        st.subheader("Policy Matrix $\pi(s)$")
        df_p1 = pd.DataFrame(st.session_state.rp_policy_matrix)
        def _style_policy1(df):
            styles = pd.DataFrame('', index=df.index, columns=df.columns)
            styles = styles.mask(df == 'OBS', 'background-color: #64748b; color: white')
            styles = styles.mask(df == 'END', 'background-color: #ef4444; color: white')
            return styles
        styled_p1 = df_p1.style.apply(_style_policy1, axis=None)
        st.dataframe(styled_p1, use_container_width=True)

if st.session_state.vi_computed:
    st.write("---")
    st.header("Value Iteration Results (Optimal Policy)")
    col_v2, col_p2 = st.columns(2)
    with col_v2:
        st.subheader("Optimal Value Matrix $V^*(s)$")
        df_v2 = pd.DataFrame(st.session_state.vi_value_matrix)
        def _style_values2(df):
            styles = pd.DataFrame('', index=df.index, columns=df.columns)
            na_mask = df.isna()
            styles = styles.mask(na_mask, 'background-color: #64748b; color: white')
            return styles
        styled_v2 = df_v2.style.apply(_style_values2, axis=None).format(na_rep="OBS", precision=2)
        st.dataframe(styled_v2, use_container_width=True)
        
    with col_p2:
        st.subheader("Optimal Policy Matrix $\pi^*(s)$")
        df_p2 = pd.DataFrame(st.session_state.vi_policy_matrix)
        def _style_policy2(df):
            styles = pd.DataFrame('', index=df.index, columns=df.columns)
            styles = styles.mask(df == 'OBS', 'background-color: #64748b; color: white')
            styles = styles.mask(df == 'END', 'background-color: #ef4444; color: white')
            return styles
        styled_p2 = df_p2.style.apply(_style_policy2, axis=None)
        st.dataframe(styled_p2, use_container_width=True)
        
    # Render Optimal Path Grid
    st.subheader("Optimal Path Visualization")
    
    # Generate custom HTML Grid
    grid_html = f'<div class="optimal-grid" style="grid-template-columns: repeat({n}, 60px);">'
    for row in range(n):
        for col in range(n):
            cell_id = row * n + col + 1
            classes = ["optimal-cell"]
            content = st.session_state.vi_policy_matrix[row][col]
            
            if cell_id == st.session_state.start_id:
                classes.append("start")
                content = f"START<br>{content}" if content != 'OBS' and content != 'END' else "START"
            elif cell_id == st.session_state.end_id:
                classes.append("end")
                content = "END"
            elif cell_id in st.session_state.obstacles:
                classes.append("obstacle")
                content = ""
            elif cell_id in st.session_state.optimal_path:
                classes.append("path")
                
            class_str = " ".join(classes)
            grid_html += f'<div class="{class_str}">{content}</div>'
    
    grid_html += '</div>'
    st.markdown(grid_html, unsafe_allow_html=True)

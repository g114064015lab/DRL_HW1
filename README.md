# Streamlit Grid Map with RL Policy Evaluation

## 🚀DEMO: https://drlhw1-8n25ygm6jljmmcugkpunwk.streamlit.app/

## Overview
This project is a modern web application built with Streamlit and Python. It allows users to generate an interactive `n x n` grid map (where `n` is between 5 and 9). Users can set up a Start cell, an End cell, and customize the grid with obstacles. 

Furthermore, the application integrates a Reinforcement Learning (RL) aspect using Iterative Policy Evaluation. Once the grid is configured, it sends the state to a Python backend solving module which evaluates random policies, finding the Value function $V(s)$ and Policy choices (arrows), to render on the frontend interface alongside the interactive grid.

## Features
- **Interactive Grid Configuration**: Dynamically build up to a 9x9 grid, visually setting Start, End, and up to `n-2` obstacles.
- **Premium UI Aesthetics**: Created using custom CSS, styled with dark/light themes, sleek glassmorphism panels, and neat state badge indicators.
- **RL Integration**: 
  - Iterative Policy Evaluation logic.
  - Generates full Value Matrix outputs based on a discount factor $\gamma=1.0$ and $-1$ rewards per step.
  - Dynamically renders policy matrices mapped onto grid cells as direction arrows.

## Project Structure
- `streamlit_app.py`: The main Streamlit application providing the interactive UI and flow.
- `rl_solver.py`: Reinforcement Learning module parsing states and calculating $V(s)$ and policies.
- `Task.md`: Development task checklist and breakdown.
- `Implementation_Plan.md`: The architectural plan mapping out the logic applied.
- `Walkthrough.md`: A visual walkthrough validating the requirements mapped against subagent testing.

## Getting Started

### Prerequisites
- Python 3.7+
- Streamlit
- Pandas

```bash
pip install streamlit pandas
```

### Running the Application

1. Open your terminal in the root directory.
2. Start the server:
```bash
streamlit run streamlit_app.py
```
*(Optionally use `python -m streamlit run streamlit_app.py` if Streamlit is not in your PATH).*

3. The application will automatically open in your default browser at `http://localhost:8501`.

### Using the App
1. Set an `n x n` size and click **Generate Square**.
2. Click an empty white space to designate the **Start Grid (Green)**.
3. Click another space to designate the **End Grid (Red)**.
4. Click up to the limit to place **Obstacles (Gray)**.
5. Once your End is placed, you may click the **Solve RL** button located in the title bar to calculate the RL matrices!

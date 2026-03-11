# Flask Grid Map with RL Policy Evaluation

## [DEMO](https://drlhw1-8n25ygm6jljmmcugkpunwk.streamlit.app/)

## Overview
This project is a modern web application built with Flask and Python. It allows users to generate an interactive `n x n` grid map (where `n` is between 5 and 9). Users can set up a Start cell, an End cell, and customize the grid with obstacles. 

Furthermore, the application integrates a Reinforcement Learning (RL) aspect using Iterative Policy Evaluation. Once the grid is configured, it sends the state to a Python backend solving module which evaluates random policies, finding the Value function $V(s)$ and Policy choices (arrows), to render on the frontend interface alongside the interactive grid.

## Features
- **Interactive Grid Configuration**: Dynamically build up to a 9x9 grid, visually setting Start, End, and up to `n-2` obstacles.
- **Premium UI Aesthetics**: Created using custom CSS, styled with dark/light themes, sleek glassmorphism panels, and neat state badge indicators.
- **RL Integration**: 
  - Iterative Policy Evaluation logic.
  - Generates full Value Matrix outputs based on a discount factor $\gamma=1.0$ and $-1$ rewards per step.
  - Dynamically renders policy matrices mapped onto grid cells as direction arrows.

## Project Structure
- `app.py`: The main Flask server application handling routes and API requests.
- `rl_solver.py`: Reinforcement Learning module parsing states and calculating $V(s)$ and policies.
- `templates/index.html`: The HTML template containing the user interface.
- `static/style.css`: The stylesheet driving the application's premium look.
- `static/script.js`: The frontend script managing interactive grid click states and API integration for the RL matrices.
- `Task.md`: Development task checklist and breakdown.
- `Implementation_Plan.md`: The architectural plan mapping out the logic applied.
- `Walkthrough.md`: A visual walkthrough validating the requirements mapped against subagent testing.

## Getting Started

### Prerequisites
- Python 3.7+
- Flask

```bash
pip install flask
```

### Running the Application

1. Open your terminal in the root directory.
2. Start the server:
```bash
python app.py
```
*(Optionally use `py app.py` on Windows if `python` is not mapped).*

3. Navigate your browser to `http://localhost:5000`.

### Using the App
1. Set an `n x n` size and click **Generate Square**.
2. Click an empty white space to designate the **Start Grid (Green)**.
3. Click another space to designate the **End Grid (Red)**.
4. Click up to the limit to place **Obstacles (Gray)**.
5. Once your End is placed, you may click the **Solve RL** button located in the title bar to calculate the RL matrices!

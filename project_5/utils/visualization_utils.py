# visualization_utils.py
"""
Utility functions for visualizing the mouse grid environment in HTML.
CSS styles should be in grid_visualization.css
"""

import numpy as np
from project_5.reinforcement_learning.mouse import (
    EMPTY, MOUSE, CHEESE, TRAP, WALL, ORGANIC_CHEESE
)


def grid_to_html(grid, title="", step_num=None, action=None, reward=None):
    """
    Convert a grid to HTML representation for display in Django template.

    Args:
        grid: 5x5 numpy array with cell values
        title: Title for this grid display
        step_num: Optional step number in trajectory
        action: Optional action taken at this step
        reward: Optional reward received

    Returns:
        HTML string representing the grid
    """
    # Symbol and color mappings
    symbols = {
        EMPTY: '·',
        MOUSE: '🐭',
        CHEESE: '🧀',
        TRAP: '⚠️',
        WALL: '⬛',
        ORGANIC_CHEESE: '🥬'  # Using lettuce emoji for organic
    }

    # Alternative text-only symbols if emojis don't work
    text_symbols = {
        EMPTY: '.',
        MOUSE: 'M',
        CHEESE: 'C',
        TRAP: 'T',
        WALL: '#',
        ORGANIC_CHEESE: 'O'
    }

    # Colors for each cell type
    colors = {
        EMPTY: '#f8f9fa',
        MOUSE: '#ffeaa7',
        CHEESE: '#fdcb6e',
        TRAP: '#ff7675',
        WALL: '#636e72',
        ORGANIC_CHEESE: '#55efc4'
    }

    html = '<div class="grid-display">'

    # Add title and step info if provided
    if title:
        html += f'<h4>{title}</h4>'

    if step_num is not None:
        html += f'<div class="step-info">'
        html += f'<span>Step {step_num}</span>'
        if action:
            html += f' | Action: <strong>{action}</strong>'
        if reward is not None:
            reward_class = 'positive' if reward > 0 else 'negative' if reward < 0 else 'neutral'
            html += f' | Reward: <span class="reward-{reward_class}">{reward:.1f}</span>'
        html += '</div>'

    # Create the grid table
    html += '<table class="game-grid">'

    for i in range(grid.shape[0]):
        html += '<tr>'
        for j in range(grid.shape[1]):
            cell_value = grid[i, j]
            symbol = symbols.get(cell_value, '?')
            color = colors.get(cell_value, '#ffffff')

            html += f'<td style="background-color: {color};">'
            html += f'<div class="grid-cell">{symbol}</div>'
            html += '</td>'
        html += '</tr>'

    html += '</table>'
    html += '</div>'

    return html


def trajectory_to_html(trajectory, max_steps=10, title="Trajectory"):
    """
    Convert a trajectory to HTML for display.

    Args:
        trajectory: Dict with 'states', 'actions', 'rewards' lists
        max_steps: Maximum number of steps to display (for brevity)
        title: Title for the trajectory

    Returns:
        HTML string representing the trajectory
    """
    states = trajectory.get('states', [])
    actions = trajectory.get('actions', [])
    rewards = trajectory.get('rewards', [])

    html = f'<div class="trajectory-display">'
    html += f'<h3>{title}</h3>'

    # Summary stats
    html += '<div class="trajectory-summary">'
    html += f'<span>Total Steps: {len(states)}</span> | '
    html += f'<span>Cheese Hits: {trajectory.get("cheese_hits", 0)}</span> | '
    html += f'<span>Organic Hits: {trajectory.get("organic_hits", 0)}</span> | '
    html += f'<span>Total Reward: {sum(rewards):.1f}</span>'
    html += '</div>'

    # Display first few steps and last step
    html += '<div class="trajectory-steps">'

    steps_to_show = min(len(states), max_steps)

    for i in range(steps_to_show):
        action = actions[i] if i < len(actions) else None
        reward = rewards[i] if i < len(rewards) else None

        html += grid_to_html(
            states[i],
            title=f"Step {i}",
            step_num=i,
            action=action,
            reward=reward
        )

    if len(states) > max_steps:
        html += f'<div class="trajectory-truncated">... {len(states) - max_steps} more steps ...</div>'

        # Show final state
        html += grid_to_html(
            states[-1],
            title="Final State",
            step_num=len(states) - 1
        )

    html += '</div>'
    html += '</div>'

    return html


def compare_trajectories_html(traj_a, traj_b, csrf_token=""):
    """
    Create side-by-side comparison of two trajectories for human preference collection.

    Args:
        traj_a, traj_b: Two trajectory dictionaries
        csrf_token: CSRF token for form submission (optional)

    Returns:
        HTML string for comparison display
    """
    html = '<div class="trajectory-comparison">'

    # Trajectory A
    html += '<div class="trajectory-option" data-trajectory="A">'
    html += '<h3>🅰️ Option A</h3>'
    html += trajectory_to_html(traj_a, max_steps=8, title="")
    html += f'''
    <form method="POST" style="margin-top: 15px;">
        <input type="hidden" name="csrfmiddlewaretoken" value="{csrf_token}">
        <input type="hidden" name="action" value="submit_preference">
        <input type="hidden" name="preference" value="A">
        <button type="submit" class="select-trajectory">
            👍 Prefer Option A
        </button>
    </form>
    '''
    html += '</div>'

    # Trajectory B
    html += '<div class="trajectory-option" data-trajectory="B">'
    html += '<h3>🅱️ Option B</h3>'
    html += trajectory_to_html(traj_b, max_steps=8, title="")
    html += f'''
    <form method="POST" style="margin-top: 15px;">
        <input type="hidden" name="csrfmiddlewaretoken" value="{csrf_token}">
        <input type="hidden" name="action" value="submit_preference">
        <input type="hidden" name="preference" value="B">
        <button type="submit" class="select-trajectory">
            👍 Prefer Option B
        </button>
    </form>
    '''
    html += '</div>'

    html += '</div>'

    return html
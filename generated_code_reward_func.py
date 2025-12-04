def reward_func(state, action, next_state):
    goal_value = ObjectTypes.goal
    x, y = next_state.agent_pos
    if next_state.grid[y][x] == goal_value:
        return 1, True
    return 0, False

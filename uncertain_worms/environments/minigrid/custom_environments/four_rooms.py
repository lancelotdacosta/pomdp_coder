# type:ignore
from __future__ import annotations

import random

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal
from minigrid.minigrid_env import MiniGridEnv


class MyFourRoomsEnv(MiniGridEnv):
    """Classic four room RL environment.

    The agent must navigate in a maze composed of four rooms. The goal is
    now restricted to appear only in the rooms specified in `goal_rooms`.

    The allowed room names are:
      - "top_left"
      - "top_right"
      - "bottom_left"
      - "bottom_right"

    For example, to spawn the goal only in the bottom right room, initialize with:

        MyFourRoomsEnv(goal_rooms=["bottom_right"])
    """

    def __init__(
        self, agent_pos=(1, 1), max_steps=100, goal_rooms: list[str] = None, **kwargs
    ):
        # Fixed agent starting position.
        self._agent_default_pos = agent_pos
        # If no goal_rooms are provided, default to bottom right.
        if goal_rooms is None:
            goal_rooms = ["bottom_right"]
        self.goal_rooms = goal_rooms

        self.size = 15
        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(
            mission_space=mission_space,
            width=self.size,
            height=self.size,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "reach the goal"

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        room_w = width // 2
        room_h = height // 2

        # Create the four rooms with connecting doors.
        for j in range(2):
            for i in range(2):
                xL = i * room_w
                yT = j * room_h
                xR = xL + room_w
                yB = yT + room_h

                # Vertical wall and door between rooms
                if i + 1 < 2:
                    self.grid.vert_wall(xR, yT, room_h)
                    # Fixed door position: middle of the wall
                    pos = (xR, yT + room_h // 2)
                    self.grid.set(*pos, None)

                # Horizontal wall and door between rooms
                if j + 1 < 2:
                    self.grid.horz_wall(xL, yB, room_w)
                    # Fixed door position: middle of the wall
                    pos = (xL + room_w // 2, yB)
                    self.grid.set(*pos, None)

        # Set fixed agent starting position.
        self.agent_pos = self._agent_default_pos
        self.grid.set(*self._agent_default_pos, None)
        self.agent_dir = 0

        # Now restrict the goal spawn to one of the allowed rooms.
        goal = Goal()
        allowed_room = random.choice(self.goal_rooms)

        # Map room names to their grid indices.
        room_mapping = {
            "top_left": (0, 0),
            "top_right": (1, 0),
            "bottom_left": (0, 1),
            "bottom_right": (1, 1),
        }

        if allowed_room not in room_mapping:
            raise ValueError(
                f"Invalid room name: {allowed_room}. Valid options are: {list(room_mapping.keys())}"
            )

        # Determine the top-left corner and size of the room's interior.
        i, j = room_mapping[allowed_room]
        top_x = i * room_w + 1
        top_y = j * room_h + 1
        size_x = room_w - 1  # interior width (excludes walls)
        size_y = room_h - 1  # interior height (excludes walls)

        self.place_obj(goal, top=(top_x, top_y), size=(size_x, size_y))

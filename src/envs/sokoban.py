import torch
from typing import Set, Tuple

class SokobanEnv:
    def __init__(self, config):
        
        self.config = config
        self.channels = self.config["channels"]
        self.action_padding_value = self.config["action_padding_value"]
        self.action_map = self.config["action_map"]

    def compile_sokoban_state(self, state, grid_shape_x, grid_shape_y):

        grid = [[" " for _ in range(grid_shape_x)] for _ in range(grid_shape_y)]

        walls, boxes, goals, player = state["walls"], state["boxes"], state["goals"], state["player"]

        for (y,x) in walls:
            grid[y][x] = "#"
        for (y,x) in goals:
            if grid[y][x] == " ":
                grid[y][x] = "."
            elif grid[y][x] == "$":
                grid[y][x] = "*"
        for (y,x) in boxes:
            if grid[y][x] == ".":
                grid[y][x] = "*"
            else:
                grid[y][x] = "$"
        py, px = player
        if grid[py][px] == ".":
            grid[py][px] = "+"
        else:
            grid[py][px] = "@"

        return "\n".join("".join(row) for row in grid)+"\n"

    def parse_sokoban_level(self, level_str: str):

        walls: Set[Tuple[int, int]] = set()
        boxes: Set[Tuple[int, int]] = set()
        goals: Set[Tuple[int, int]] = set()
        
        player = None

        rows = level_str.split("\n")

        for r, row in enumerate(rows):
            for c, ch in enumerate(row):
                if ch == "#":
                    walls.add((r, c))

                elif ch == "$":
                    boxes.add((r, c))

                elif ch == ".":
                    goals.add((r, c))

                elif ch == "@":
                    player = (r, c)

                elif ch == "*":          # box on goal
                    boxes.add((r, c))
                    goals.add((r, c))

                elif ch == "+":          # player on goal
                    player = (r, c)
                    goals.add((r, c))

        if player is None:
            raise ValueError("No player found in level")

        return {
            "walls": walls,
            "boxes": boxes,
            "goals": goals,
            "player": player
        }   
    
    def step(self, state, action_str):

        walls, boxes, goals, player = state["walls"], state["boxes"], state["goals"], state["player"]

        dy, dx = self.action_map[int(action_str)]
        new_pos_player = (player[0]+dy, player[1]+dx)
        if new_pos_player in walls:
            return None, "player hit wall", 0.0

        reward = -0.01  # Small penalty per step

        if new_pos_player in boxes:
            new_pos_box = (new_pos_player[0]+dy, new_pos_player[1]+dx)
            if new_pos_box in boxes:
                return None, "box hit box", 0.0

            if new_pos_box in walls:
                return None, "box hit wall", 0.0

            # Check if box was on goal before moving
            was_on_goal = new_pos_player in goals
            # Check if box will be on goal after moving
            will_be_on_goal = new_pos_box in goals

            boxes.remove(new_pos_player)
            boxes.add(new_pos_box)

            # Update reward based on box movement
            if will_be_on_goal and not was_on_goal:
                reward += 1.0  # Pushed box onto goal
            elif was_on_goal and not will_be_on_goal:
                reward += -1.0  # Pushed box off goal

        player = new_pos_player
        if tuple(sorted(boxes)) == tuple(sorted(goals)):
            status = "solved"
            reward += 10.0  # Bonus for solving the puzzle
        else:
            status = "in progress"

        new_state = {"walls": walls, "boxes": boxes, "goals": goals, "player": player}

        return new_state, status, reward
    
    def play(self, state, actions_str, early_stop=False):
        states = [state]
        total_reward = 0.0
        if len(actions_str) == 0:
            return states, "no actions", total_reward
        else:
            for action_str in actions_str:
                state, status, reward = self.step(state, action_str)
                total_reward += reward
                if status in ["solved", "in progress"]:
                    states.append(state)
                    if early_stop and status == "solved":
                        break
                else: break

        return states, status, total_reward


    def symbolic_state_to_tensor(self, state, grid_shape_x, grid_shape_y):
        
        num_channels = len(self.channels)
        tensor = torch.zeros((grid_shape_x, grid_shape_y, num_channels), dtype=torch.float32)

        for c, key in enumerate(self.channels):
            values = state[key] if key != "player" else {state[key]}
            idx = torch.tensor(list(values), dtype=torch.long)  # shape [N, 2]
            tensor[idx[:, 0], idx[:, 1], c] = 1.0

        return tensor

    def step_batch(self, batch, policies, temperature=1.0):
        """
        policies: [B, A] or [B, T, A] (last timestep assumed)
        """

        states = batch["states"]
        device = batch["states_tensors"].device
        B = len(states)

        H = batch["states_tensors"].size(-3)
        W = batch["states_tensors"].size(-2)

        logits = policies / temperature

        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()  # [B]

        new_states = []
        new_states_tensor = []
        new_attention_mask = []
        rewards = []

        for b in range(B):
            state = states[b]

            # if already terminated then pad
            if batch["attention_mask"][b, -1] == 0:
                new_states.append(state)
                new_states_tensor.append(
                    torch.zeros_like(batch["states_tensors"][b, :1]).to(device)
                )
                new_attention_mask.append([0])
                rewards.append(0.0)
                continue

            action_str = str(actions[b].item())
            new_state, status, reward = self.step(state, action_str)

            if status == "in progress":
                obs = self.symbolic_state_to_tensor(new_state, H, W).unsqueeze(0)
                mask = 1
            else:
                obs = torch.zeros_like(batch["states_tensors"][b, :1])
                mask = 0

            new_states.append(new_state)
            new_states_tensor.append(obs.to(device))
            new_attention_mask.append([mask])
            rewards.append(reward)

        batch["states"] = new_states
        new_states_tensor = torch.stack(new_states_tensor, dim=0)
        new_attention_mask = torch.tensor(new_attention_mask, device=device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)

        return new_states_tensor, new_attention_mask, rewards_tensor

    def symbolic_batch_to_tensor(self, states, grid_shape_x, grid_shape_y, device=None):
        """
        states: list of symbolic states (len B)
        returns: [B, H, W, C]
        """

        B = len(states)
        C = len(self.channels)

        tensor = torch.zeros(
            (B, grid_shape_x, grid_shape_y, C),
            dtype=torch.float32,
            device=device
        )

        for b, state in enumerate(states):
            for c, key in enumerate(self.channels):
                values = state[key] if key != "player" else {state[key]}
                if len(values) == 0:
                    continue
                idx = torch.tensor(list(values), dtype=torch.long, device=device)
                tensor[b, idx[:, 0], idx[:, 1], c] = 1.0

        return tensor

    def collate_fn(self, batch):

        batch_states = [item["states_tensor"] for item in batch]
        batch_actions = [item["actions_id"] for item in batch]

        batch_attention_mask = [
            torch.tensor([1]*len(item["actions_id"])) for item in batch
        ]

        batch_attention_mask = torch.nn.utils.rnn.pad_sequence(batch_attention_mask, batch_first=True, padding_value=0)
        batch_states_padded = torch.nn.utils.rnn.pad_sequence(batch_states, batch_first=True, padding_value=0.0)
        batch_actions_padded = torch.nn.utils.rnn.pad_sequence(batch_actions, batch_first=True, padding_value=self.action_padding_value)

        level_strs = [item["level_str"] for item in batch]
        action_strs = [item["actions_str"] for item in batch]
        folder_names = [item["folder_name"] for item in batch]
        level_names = [item["level_name"] for item in batch]

        return {
            "level_strs": level_strs,
            "action_strs": action_strs,
            "folder_names": folder_names,
            "level_names": level_names,
            "states_tensors": batch_states_padded,
            "actions_ids": batch_actions_padded,
            "attention_mask": batch_attention_mask
        }

    def get_dynamic_batch(self, level_strs):

        dynamic_batch = {}

        states = [self.parse_sokoban_level(level_str) for level_str in level_strs]
        states_tensor = torch.stack([
            self.symbolic_state_to_tensor(state, 10, 10).unsqueeze(0)
            for state in states
        ])
        batch_attention_mask = torch.tensor([[1]*1 for _ in range(len(level_strs))])  # Dummy attention mask for 1 think step
        dynamic_batch["states_0"] = states
        dynamic_batch["states_tensors"] = states_tensor
        dynamic_batch["attention_mask"] = batch_attention_mask
        
        return dynamic_batch
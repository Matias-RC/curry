"""
Boxoban Env,
Custom SubProcVecEnv,
Boxoban Level bandit
"""
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from gym_sokoban.envs import SokobanEnv
import random
import uuid
from typing import List, Dict, Tuple, Optional
from gym_sokoban.envs.boxoban_env import BoxobanEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from gym_sokoban.envs.render_utils import room_to_rgb
import os
from os import listdir
from os.path import isfile, join
import requests
import zipfile
from tqdm import tqdm

class BoxobanCustomReset(BoxobanEnv):
    def __init__(self, max_steps=120, difficulty='unfiltered', split='train'):
        super().__init__(max_steps, difficulty, split)
    

    def _save_level_state(self):
        self.saved_state = {
            "room_fixed": self.room_fixed.copy(),
            "room_state": self.room_state.copy(),
            "box_mapping": self.box_mapping.copy() if hasattr(self, "box_mapping") else None
        }
    
    def _restore_level(self):
        state:Dict[str,dict ] = self.saved_state

        self.room_fixed = state["room_fixed"].copy()
        self.room_state = state["room_state"].copy()
        if state['box_mapping'] is not None:
            self.box_mapping = state['box_mapping'].copy()
        
        self.player_position  = np.argwhere(self.room_state == 5)[0]
        self.num_env_steps = 0
        self.reward_last = 0
        self.boxes_on_target = 0 
    
    def select_room(self, _id=None):
            if _id is None:
                return super().select_room()
                
            source = _id["source"]
            item = _id["item"]
            source_file = join(self.train_data_dir, source)

            this_map = []
            found_id = False

            with open(source_file, "r") as sf:
                for line in sf:
                    clean_line = line.strip()
                    
                    if clean_line == f"; {item}":
                        found_id = True
                        continue
                    
                    if found_id:
                        if clean_line.startswith(";"):
                            break 
                        if clean_line.startswith("#"):
                            this_map.append(clean_line)
                
            if not this_map:
                raise ValueError(f"Level {item} not found in {source_file}")
            
            self.room_fixed, self.room_state, self.box_mapping = self.generate_room(this_map)
            
    def reset(self, seed=None, options=None):
            self.cache_path = '.sokoban_cache'
            self.train_data_dir = os.path.join(self.cache_path, 'boxoban-levels-master', self.difficulty, self.split)

            if not os.path.exists(self.cache_path):
                url = "https://github.com/deepmind/boxoban-levels/archive/master.zip"
                if getattr(self, 'verbose', False):
                    print(f'Boxoban: Pregenerated levels not downloaded. Starting from "{url}"')
                
                response = requests.get(url, stream=True)
                if response.status_code != 200:
                    raise Exception(f"Could not download levels from {url}")

                os.makedirs(self.cache_path)
                path_to_zip_file = os.path.join(self.cache_path, 'boxoban_levels-master.zip')
                
                with open(path_to_zip_file, 'wb') as handle:
                    for data in tqdm(response.iter_content()):
                        handle.write(data)

                with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
                    zip_ref.extractall(self.cache_path)
            
            if options is None:
                options = {"hard_reset": True, "id": None}

            if options.get("hard_reset", True) or self.saved_state is None:
                self.select_room(options.get("id"))
                self._save_level_state()
            else:
                self._restore_level()

            self.num_env_steps = 0
            self.reward_last = 0
            self.boxes_on_target = 0

            starting_observation = room_to_rgb(self.room_state, self.room_fixed)

            return starting_observation, {}
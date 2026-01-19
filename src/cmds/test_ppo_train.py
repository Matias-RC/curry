import torch
import torch.nn as nn
import torch.nn.functional as F
import sys


levels = [
    "##########\n"
    "#@ #######\n"
    "#.$#######\n"
    "#   ######\n"
    "#.$ ######\n"
    "#  #######\n"
    "# $ ######\n"
    "#$ . #####\n"
    "# .  #####\n"
    "##########",

    "##########\n"
    "##########\n"
    "# ###@####\n"
    "#    $   #\n"
    "#   $    #\n"
    "# ##  ####\n"
    ".##  ####\n"
    "# ###$$.##\n"
    "#  .   . #\n"
    "##########",

    "##########\n"
    "#####   ##\n"
    "#####   ##\n"
    "####.    #\n"
    "# .  $@ ##\n"
    "# $ $ $ ##\n"
    "#   .   ##\n"
    "#####.   #\n"
    "######   #\n"
    "##########",

    "##########\n"
    "######## #\n"
    "###      #\n"
    "### $ . .#\n"
    "### $ #  #\n"
    "#@$  ##  #\n"
    "##  . #  #\n"
    "## $ ##  #\n"
    "## .     #\n"
    "##########",
]


def main():
    sys.path.append("./src")

    from data.dataset import Dataset
    from envs.sokoban import SokobanEnv
    config_sokoban_env = {
        "action_padding_value": -100,
        "channels": ['boxes', 'goals', 'player', 'walls'],
        "action_map": [(-1,0),(0,1),(1,0),(0,-1)],
    }
    sokoban_env = SokobanEnv(config_sokoban_env)

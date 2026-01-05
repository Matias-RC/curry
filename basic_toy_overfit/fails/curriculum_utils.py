import random

def apply_midway_curriculum(env, episode_idx):
    """
    Modifies env in-place depending on episode index.
    Assumes env.initialize_state() was already called.
    """
    y = random.randint(2, 4)
    pos = (y, 10)

    # restore previous lock position to walls
    old_pos = env.lock_entries[0].pos
    env.walls[0].add(old_pos)
    env.walls[1].add(old_pos)

    # remove new position from walls
    env.walls[0].remove(pos)
    env.walls[1].remove(pos)

    env.lock_entries[0].pos = pos

    # Key placement curriculum
    if episode_idx < 80:
        env.lock_entries[env.current_key].my_key.pos = (y, 9)
        env.lock_entries[env.current_key].next_key.original_pos = (y, 7)
        env.lock_entries[env.current_key].next_key.pos = (y, 7)
    elif episode_idx < 160:
        dy = random.choice([-1, 0, 1])
        x = random.choice([8, 9])
        env.lock_entries[env.current_key].my_key.pos = (y + dy, x)
        env.lock_entries[env.current_key].next_key.original_pos = (y+ dy, 7)
        env.lock_entries[env.current_key].next_key.pos = (y+ dy, 7)
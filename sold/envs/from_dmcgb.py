import numpy as np

import dm_control
from dm_control.suite.wrappers import action_scale


import envs.from_dmcontrol as dmc
# import envs.dmcgb.dmcgb_geometric as dmcgb_geometric
import envs.wrappers.dmcgb_photometric as dmcgb_photometric


dmcgb_photometric_modes = dmcgb_photometric.dmcgb_photometric_modes
# dmcgb_geometric_modes = dmcgb_geometric.dmcgb_geometric_modes
valid_modes = dmcgb_photometric_modes #+ dmcgb_geometric_modes


def make_env(name, image_size, max_episode_steps, action_repeat, seed, mode="color_hard"):
    domain_name, task_name = name.split('_', 1)
    domain_name = dict(cup='ball_in_cup').get(domain_name, domain_name)
    assert mode in valid_modes , f'Specified mode "{mode}" is not supported'

    env = dm_control.suite.load(domain_name, task_name, task_kwargs={'random': seed}, visualize_reward=False)
    env._domain_name = domain_name
    # Wrappers
    env = dmc.ActionDTypeWrapper(env, np.float32)
    env = dmc.ActionRepeatWrapper(env, action_repeat)
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
    # ------ DMCGB ------ #
    # env = dmcgb_geometric.ShiftWrapper(env, mode, seed) 
    # env = dmcgb_geometric.RotateWrapper(env, mode, seed) 
    env = dmcgb_photometric.ColorVideoWrapper(env, mode, seed, video_render_size=256) 
    # ------------------- #
    env = dmc.ExtendedTimeStepWrapper(env)
    env = dmc.TimeStepToGymWrapper(env, domain_name, task_name)
    env = dmc.TimeLimit(env, max_episode_steps)
    env = dmc.Pixels(env, image_size)
    return env


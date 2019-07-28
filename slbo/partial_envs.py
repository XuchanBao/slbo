# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
# from slbo.envs.bm_envs.gym.half_cheetah import HalfCheetahEnv
# from slbo.envs.bm_envs.gym.walker2d import Walker2dEnv
# from slbo.envs.mujoco.humanoid_env import HumanoidEnv
# from slbo.envs.bm_envs.gym.ant import AntEnv
# from slbo.envs.bm_envs.gym.hopper import HopperEnv
# from slbo.envs.bm_envs.gym.swimmer import SwimmerEnv
# from slbo.envs.bm_envs.gym.reacher import ReacherEnv
# from slbo.envs.bm_envs.gym.pendulum import PendulumEnv
# from slbo.envs.bm_envs.gym.inverted_pendulum import InvertedPendulumEnv
# from slbo.envs.bm_envs.gym.acrobot import AcrobotEnv
from slbo.envs.bm_envs.gym.roboschool_reacher import RoboschoolReacher
from slbo.envs.bm_envs.gym.roboschool_cheetah import RoboschoolHalfCheetah, RoboschoolAnt
from slbo.envs.bm_envs.gym.pybullet_reacher import ReacherBulletEnv


def make_env(id: str):
    envs = {
        'RoboschoolReacher': RoboschoolReacher,
        'PybulletReacher': ReacherBulletEnv,
        'roboschoolHalfCheetah': RoboschoolHalfCheetah,
        'roboschoolAnt': RoboschoolAnt
    }
    print(id)
    env = envs[id]()
    if not hasattr(env, 'reward_range'):
        env.reward_range = (-np.inf, np.inf)
    if not hasattr(env, 'metadata'):
        env.metadata = {}
    env.seed(np.random.randint(2**60))
    return env

from pybulletgym.envs.mujoco.envs.env_bases import BaseBulletEnv
from pybulletgym.envs.mujoco.robots.manipulators.reacher import Reacher
from pybulletgym.envs.mujoco.scenes.scene_bases import SingleRobotEmptyScene
import numpy as np
from slbo.envs import BaseModelBasedEnv


class ReacherBulletEnv(BaseBulletEnv, BaseModelBasedEnv):

    def __init__(self):
        self.robot = Reacher()
        BaseBulletEnv.__init__(self, self.robot)

    def create_single_player_scene(self, bullet_client):
        return SingleRobotEmptyScene(bullet_client, gravity=0.0, timestep=0.0165, frame_skip=1)

    def _step(self, a):
        assert (not self.scene.multiplayer)
        reward = -self.cost_np_vec(self._old_state, a, self._old_state * 0.0)
        self.robot.apply_action(a)
        self.scene.global_step()

        state = self.robot.calc_state()  # sets self.to_target_vec
        potential_old = self.potential
        self.potential = self.robot.calc_potential()

        stuck_joint_cost = -0.1 if np.abs(np.abs(self.robot.gamma) - 1) < 0.01 else 0.0
        self.HUD(state, a, False)
        self._old_state = state
        return state, reward, False, {}

    def camera_adjust(self):
        x, y, z = self.robot.fingertip.pose().xyz()
        x *= 0.5
        y *= 0.5
        self.camera.move_and_look_at(0.3, 0.3, 0.3, x, y, z)

    def mb_step(self, states, actions, next_states):
        # returns rewards and dones
        # forward rewards are calculated based on states, instead of next_states as in original SLBO envs
        if getattr(self, 'action_space', None):
            actions = np.clip(actions, self.action_space.low,
                              self.action_space.high)
        rewards = - self.cost_np_vec(states, actions, next_states)
        return rewards, np.zeros_like(rewards, dtype=np.bool)

    def cost_np_vec(self, obs, acts, next_obs):
        dist_vec = obs[:, -3:]
        reward_dist = - np.linalg.norm(dist_vec, axis=1)
        reward_ctrl = - np.sum(np.square(acts), axis=1)
        reward = reward_dist + reward_ctrl
        return -reward

from roboschool.scene_abstract import SingleRobotEmptyScene
from roboschool.gym_mujoco_xml_env import RoboschoolMujocoXmlEnv
import gym, gym.spaces, gym.utils, gym.utils.seeding
import numpy as np
import os, sys
from slbo.envs import BaseModelBasedEnv

class RoboschoolReacher(RoboschoolMujocoXmlEnv, BaseModelBasedEnv):
    '''
    Get the end of two-link robotic arm to a given spot.
    Similar to MuJoCo reacher. 
    '''
    def __init__(self):
        RoboschoolMujocoXmlEnv.__init__(self, 'reacher.xml', 'body0', action_dim=2, obs_dim=9)

    def create_single_player_scene(self):
        return SingleRobotEmptyScene(gravity=0.0, timestep=0.0165, frame_skip=1)

    TARG_LIMIT = 0.27
    def robot_specific_reset(self):
        self.jdict["target_x"].reset_current_position(self.np_random.uniform( low=-self.TARG_LIMIT, high=self.TARG_LIMIT ), 0)
        self.jdict["target_y"].reset_current_position(self.np_random.uniform( low=-self.TARG_LIMIT, high=self.TARG_LIMIT ), 0)
        self.fingertip = self.parts["fingertip"]
        self.target    = self.parts["target"]
        self.central_joint = self.jdict["joint0"]
        self.elbow_joint   = self.jdict["joint1"]
        self.central_joint.reset_current_position(self.np_random.uniform( low=-3.14, high=3.14 ), 0)
        self.elbow_joint.reset_current_position(self.np_random.uniform( low=-3.14, high=3.14 ), 0)

    def apply_action(self, a):
        assert( np.isfinite(a).all() )
        self.central_joint.set_motor_torque( 0.05*float(np.clip(a[0], -1, +1)) )
        self.elbow_joint.set_motor_torque( 0.05*float(np.clip(a[1], -1, +1)) )

    def calc_state(self):
        theta,      self.theta_dot = self.central_joint.current_relative_position()
        self.gamma, self.gamma_dot = self.elbow_joint.current_relative_position()
        target_x, _ = self.jdict["target_x"].current_position()
        target_y, _ = self.jdict["target_y"].current_position()
        self.to_target_vec = np.array(self.fingertip.pose().xyz()) - np.array(self.target.pose().xyz())
        return np.array([
            target_x,
            target_y,
            self.to_target_vec[0],
            self.to_target_vec[1],
            np.cos(theta),
            np.sin(theta),
            self.theta_dot,
            self.gamma,
            self.gamma_dot,
        ])

    def calc_potential(self):
        return -1.0 * np.linalg.norm(self.to_target_vec)

    def step(self, a):
        old_state = self.calc_state()  # sets self.to_target_vec
        assert(not self.scene.multiplayer)
        self.apply_action(a)
        self.scene.global_step()

        state = self.calc_state()  # sets self.to_target_vec

        potential_old = self.potential
        self.potential = self.calc_potential()

        self.rewards = -self.cost_np_vec(old_state[None, :], a[None, :], state[None, :])[0]
        self.frame += 1
        self.done += 0
        # self.reward += sum(self.rewards)
        # self.HUD(state, a, False)
        return state, self.rewards, False, {}

    def camera_adjust(self):
        x, y, z = self.fingertip.pose().xyz()
        x *= 0.5
        y *= 0.5
        self.camera.move_and_look_at(0.3, 0.3, 0.3, x, y, z)

    def mb_step(self, states, actions, next_states):
        rewards = - self.cost_np_vec(states, actions, next_states)
        return rewards, np.zeros_like(rewards, dtype=np.bool)

    def cost_np_vec(self, obs, acts, next_obs):
        dist_vec = obs[:, 2:5]
        dist_vec[:, -1] = 0.0
        reward_dist = - np.linalg.norm(dist_vec, axis=1)
        reward_ctrl = - np.sum(np.square(acts), axis=1)
        return -(reward_dist + reward_ctrl)

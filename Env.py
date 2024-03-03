#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import random
import cv2
import os

import gym
from gym import spaces



class Multi_Agent_SoccerEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, n_agents=2, m_agents=2, field_width=640, field_height=480, goal_width = 100):
        super(Multi_Agent_SoccerEnv, self).__init__()

        self.n_agents = n_agents
        self.m_agents = m_agents
        self.total_agents = n_agents + m_agents
        self.field_width = field_width
        self.field_height = field_height
        self.goal_width = goal_width
        self.parameter = 1
        
        self.speed = 1.
        self.movements = [
            np.array([0, self.speed]),  # Up
            np.array([0, -self.speed]),  # Down
            np.array([self.speed, 0]),  # Right
            np.array([-self.speed, 0]),  # Left
            np.array([self.speed, self.speed]),  # Diagonal up-right
            np.array([-self.speed, self.speed]),  # Diagonal up-left
            np.array([self.speed, -self.speed]),  # Diagonal down-right
            np.array([-self.speed, -self.speed])  # Diagonal down-left
        ]
        
        self.opponent_patterns = [random.choice(self.movements) for _ in range(m_agents)]
        self.pattern_switch_timer = 100  # Switch pattern every 100 steps
        self.step_counter = 0
        
        
        self.goal_position = np.array([20, self.field_height/2])  
        self.ball_carrier_index = np.random.randint(m_agents)  # Randomly select the initial ball carrier

        # Action space for all agents concatenated into one vector
        self.action_space = spaces.Box(low=-1, high=1, shape=(2*self.n_agents,), dtype=np.float32)

        # Observation space: positions (x, y) and velocities (vx, vy) of both agents
        self.observation_space = spaces.Box(low=np.array([0, 0] * self.total_agents + [-field_width, -field_height] * self.total_agents),
                                            high=np.array([field_width, field_height] * self.total_agents + [field_width, field_height] * self.total_agents),
                                            dtype=np.float32)
        

        # Initialize positions and velocities for all agents
        self.player_positions = np.random.uniform(low=[self.field_width/16, self.field_height/4], high=[self.field_width/4, 3*self.field_height/4], size=(self.n_agents, 2))
        self.player_positions = self.player_positions.flatten()
        self.opponent_positions = np.random.uniform(low=[self.field_width/2, 0], high=[self.field_width, self.field_height], size=(self.m_agents, 2))
        self.opponent_positions = self.opponent_positions.flatten()
        
        self.player_velocities = np.zeros((self.n_agents, 2))
        self.player_velocities = self.player_velocities.flatten()
        self.opponent_velocities = np.zeros((self.m_agents, 2))
        self.opponent_velocities = self.opponent_velocities.flatten()

        self.init_field()

    def init_field(self):
        self.field = np.zeros((self.field_height, self.field_width, 3), np.uint8)
        self.field[:] = (0, 255, 0)
        cv2.rectangle(self.field, (10, 10), (self.field_width - 10, self.field_height - 10), (255, 255, 255), 2)
        
        goal_y_start = self.field_height // 2 - self.goal_width // 2
        goal_y_end = goal_y_start + self.goal_width
        cv2.rectangle(self.field, (0, goal_y_start), (10, goal_y_end), (255, 255, 255), -1)
        cv2.rectangle(self.field, (self.field_width - 10, goal_y_start), (self.field_width, goal_y_end), (255, 255, 255), -1)
        
        cv2.line(self.field, (self.field_width // 2, 0), (self.field_width // 2, self.field_height), (255, 255, 255), 2)

        center_circle_radius = 50
        center_of_field = (self.field_width // 2, self.field_height // 2)
        cv2.circle(self.field, center_of_field, center_circle_radius, (255, 255, 255), 2)
        
        goal_box_width = 100
        goal_box_height = 250
        
        s_goal_box_width = 50
        s_goal_box_height = 150

        cv2.rectangle(self.field, (10, self.field_height // 2 - goal_box_height // 2),
                      (10 + goal_box_width, self.field_height // 2 + goal_box_height // 2), (255, 255, 255), 2)

        cv2.rectangle(self.field, (self.field_width - 10 - goal_box_width, self.field_height // 2 - goal_box_height // 2),
                      (self.field_width - 10, self.field_height // 2 + goal_box_height // 2), (255, 255, 255), 2)
        
        cv2.rectangle(self.field, (10, self.field_height // 2 - s_goal_box_height // 2),
                      (10 + s_goal_box_width, self.field_height // 2 + s_goal_box_height // 2), (255, 255, 255), 2)

        cv2.rectangle(self.field, (self.field_width - 10 - s_goal_box_width, self.field_height // 2 - s_goal_box_height // 2),
                      (self.field_width - 10, self.field_height // 2 + s_goal_box_height // 2), (255, 255, 255), 2)
        
        
        penalty_spot_distance = 75
        penalty_spot_diameter = 8

        semi_circle_radius = 50

        penalty_spot_left = (10 + penalty_spot_distance, self.field_height // 2)
        penalty_spot_right = (self.field_width - 10 - penalty_spot_distance, self.field_height // 2)

        cv2.circle(self.field, penalty_spot_left, penalty_spot_diameter // 2, (255, 255, 255), -1)
        cv2.circle(self.field, penalty_spot_right, penalty_spot_diameter // 2, (255, 255, 255), -1)

        cv2.ellipse(self.field, penalty_spot_left, (semi_circle_radius, semi_circle_radius), 180, 120, 240, (255, 255, 255), 2)
        cv2.ellipse(self.field, penalty_spot_right, (semi_circle_radius, semi_circle_radius), 0, 120, 240, (255, 255, 255), 2)


    
    def check_for_collisions(self):
        # Assume each agent is represented as a circle for collision detection
        agent_radius = 15

        # Check collisions between all pairs of agents
        for i in range(self.total_agents):
            for j in range(i + 1, self.total_agents):
                # Calculate distance between agents
                if i < self.n_agents:
                    position_i = self.player_positions[2*i:2*i+2]
                else:
                    index_i = i - self.n_agents
                    position_i = self.opponent_positions[2*index_i:2*index_i+2]

                if j < self.n_agents:
                    position_j = self.player_positions[2*j:2*j+2]
                else:
                    index_j = j - self.n_agents
                    position_j = self.opponent_positions[2*index_j:2*index_j+2]

                distance = np.linalg.norm(position_i - position_j)

                if distance < 2 * agent_radius:
                    overlap = 2 * agent_radius - distance
                    if distance > 0:
                        direction = (position_i - position_j) / distance
                    else:
                        direction = np.array([1, 0])

                    if i < self.n_agents:
                        self.player_positions[2*i:2*i+2] += direction * (overlap / 2)
                    else:
                        self.opponent_positions[2*index_i:2*index_i+2] += direction * (overlap / 2)

                    if j < self.n_agents:
                        self.player_positions[2*j:2*j+2] -= direction * (overlap / 2)
                    else:
                        self.opponent_positions[2*index_j:2*index_j+2] -= direction * (overlap / 2)
                      
                        
    def pass_ball(self):
        pass_threshold = 50

        closest_distance_to_defender = min(
            np.linalg.norm(self.player_positions[2*i:2*i+2] - self.opponent_positions[2*self.ball_carrier_index:2*self.ball_carrier_index+2])
            for i in range(self.n_agents)
        )

        if self.m_agents > 1:
            if closest_distance_to_defender < pass_threshold:
                best_teammate_index = self.find_best_pass_target()
                self.ball_carrier_index = best_teammate_index
            

    def find_best_pass_target(self):
        goal_position = self.goal_position
        best_teammate_index = None
        best_score = -float('inf')

        for i in range(self.m_agents):
            if i == self.ball_carrier_index:
                continue

            teammate_position = self.opponent_positions[2*i:2*i+2]
            distance_to_goal = np.linalg.norm(teammate_position - goal_position)

            min_distance_to_defender = min(
                np.linalg.norm(teammate_position - self.player_positions[2*j:2*j+2])
                for j in range(self.n_agents)
            )

            score = min_distance_to_defender - distance_to_goal

            if score > best_score:
                best_score = score
                best_teammate_index = i

        return best_teammate_index
                

    def calculate_evasion(self, opponent_index):
        evasion_vector = np.array([0.0, 0.0])
        avoidance_threshold = 50

        current_opponent_position = self.opponent_positions[2*opponent_index:2*opponent_index+2]

        for j in range(self.n_agents):
            player_position = self.player_positions[2*j:2*j+2]
            distance_to_player = np.linalg.norm(current_opponent_position - player_position)
            if distance_to_player < avoidance_threshold:
                direction_away_from_player = current_opponent_position - player_position
                if np.linalg.norm(direction_away_from_player) > 0:
                    evasion_vector += (direction_away_from_player / np.linalg.norm(direction_away_from_player)) * (1 / distance_to_player)

        for j in range(self.m_agents):
            if j != opponent_index:
                other_opponent_position = self.opponent_positions[2*j:2*j+2]
                distance_to_opponent = np.linalg.norm(current_opponent_position - other_opponent_position)
                if distance_to_opponent < avoidance_threshold:
                    direction_away_from_opponent = current_opponent_position - other_opponent_position
                    if np.linalg.norm(direction_away_from_opponent) > 0:
                        evasion_vector += (direction_away_from_opponent / np.linalg.norm(direction_away_from_opponent)) * (1 / distance_to_opponent)

        return evasion_vector
    
    
    def opponent_strategy(self):
        for i in range(self.m_agents):
            if i == self.ball_carrier_index:
                direction_to_goal = self.goal_position - self.opponent_positions[2*i:2*i+2]
                normalized_direction_to_goal = direction_to_goal / np.linalg.norm(direction_to_goal + 1e-8)

                evasion_direction = self.calculate_evasion(i)
                evasion_scale = min(np.linalg.norm(evasion_direction) / 100, 1)
                scaled_evasion_direction = evasion_direction * evasion_scale

                movement_direction = normalized_direction_to_goal + scaled_evasion_direction
                normalized_movement_direction = movement_direction / (np.linalg.norm(movement_direction) + 1e-8)

                self.opponent_positions[2*i:2*i+2] += self.parameter * normalized_movement_direction * self.speed
            else:
                evasion_direction = self.calculate_evasion(i)
                normalized_evasion = evasion_direction / np.linalg.norm(evasion_direction) if np.linalg.norm(evasion_direction) > 0 else np.zeros(2)
                self.opponent_positions[2*i:2*i+2] += self.parameter * normalized_evasion * self.speed + self.opponent_patterns[i]


        
        
    def step(self, action):
        done = False
        reward = 0
        
        for i in range(self.n_agents):
            self.player_velocities[2*i:2*i+2] = np.clip(action[2*i:2*i+2], -1, 1)
            self.player_positions[2*i:2*i+2] += self.parameter * self.player_velocities[2*i:2*i+2]


        self.opponent_strategy()

        self.pass_ball()
            
        for i in range(self.n_agents):    
            self.player_positions[2*i:2*i+2] = self._check_boundaries(self.player_positions[2*i:2*i+2])
        for i in range(self.m_agents):
            self.opponent_positions[2*i:2*i+2] = self._check_boundaries(self.opponent_positions[2*i:2*i+2])

        self.check_for_collisions()
        
        
        """
        Reward function
        You can define your reward function here depending on the behavior you are looking for
        
        """
        
    
    
    
        return self._get_state(), reward, done, {}
    

    def reset(self):
        # Reset positions and velocities of all agents
        self.player_positions = np.random.uniform(low=[self.field_width/16, self.field_height/4], high=[self.field_width/4, 3*self.field_height/4], size=(self.n_agents, 2))
        self.player_positions = self.player_positions.flatten()
        self.opponent_positions = np.random.uniform(low=[self.field_width/2, 0], high=[self.field_width, self.field_height], size=(self.m_agents, 2))
        self.opponent_positions = self.opponent_positions.flatten()
        
        self.player_velocities = np.zeros((self.n_agents, 2))
        self.player_velocities = self.player_velocities.flatten()
        self.opponent_velocities = np.zeros((self.m_agents, 2))
        self.opponent_velocities = self.opponent_velocities.flatten()
        
        return self._get_state()
    
    
    def _get_state(self):
        # Combine positions and velocities of all agents into the state vector
        return np.concatenate([self.player_positions.flatten(), self.opponent_positions.flatten(),
                               self.player_velocities.flatten(), self.opponent_velocities.flatten()])
    
    def _check_boundaries(self, positions):
        positions = np.clip(positions, [30, 30], [self.field_width - 30, self.field_height - 30])
        return positions
    
    

    def render(self, mode='human'):
        """
        I found this comment important to identify your players
        """
        if mode == 'human':
            self.init_field()

        for i in range(self.n_agents):
            color = (0, 0, 0)  # Black color for your agents
            cv2.circle(self.field, tuple(self.player_positions[2*i:2*i+2].astype(int)), 15, color, -1)

        for j in range(self.m_agents):
            if j == self.ball_carrier_index:
                color = (255, 0, 0)  # Blue color for the ball carrier
            else:
                color = (0, 0, 255)  # Red color for other opponent agents
            cv2.circle(self.field, tuple(self.opponent_positions[2*j:2*j+2].astype(int)), 15, color, -1)
            

        cv2.imshow('Soccer Field', self.field)
        cv2.waitKey(1)



    def close(self):
        cv2.destroyAllWindows()

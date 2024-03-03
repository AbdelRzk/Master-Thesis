#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from Env import Multi_Agent_SoccerEnv



# Usage example
env = Multi_Agent_SoccerEnv(n_agents=2, m_agents=2)

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
act_limit = env.action_space.high[0]

print('---------------------------------------')
print('State dimension:', obs_dim)
print('Action dimension:', act_dim)
print('Action limit:', act_limit)
print('---------------------------------------')


obs = env.reset()
done = False

while True:
    actions = env.action_space.sample()
    obs, reward, done, info = env.step(actions)
    env.render()
    key = cv2.waitKey(1)
    
    if key == ord("q"):
        break
env.close()


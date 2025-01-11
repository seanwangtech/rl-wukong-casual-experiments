import gymnasium as gym
env = gym.make("ALE/Kaboom-v5", render_mode='rgb_array')
import cv2

done = True
import time
t1 = time.time()
pre_step = 0
total_reward = 0
for step in range(5000):
    if done:
        state,_ = env.reset()
    state, reward, done,trancated, info = env.step(env.action_space.sample())
    total_reward += reward
    if(info['frame_number']%5==0):
        cv2.imshow('image', cv2.resize(cv2.cvtColor(state, cv2.COLOR_RGB2BGR), (240, 240), interpolation=cv2.INTER_CUBIC)) # cv2.imshow('image', state)
        # time.sleep(0.1)
        cv2.waitKey(1)
    if time.time() - t1 > 1:
        print(f'FPS: {step - pre_step}, steps: {step}, reward: {total_reward}, state.shape: {state.shape}')
        # cv2.waitKey(1)
        pre_step = step
        t1 = time.time()
    env.render()

env.close()
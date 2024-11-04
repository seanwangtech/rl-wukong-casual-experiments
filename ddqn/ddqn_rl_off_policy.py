import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
from wukong_env import BlackMythWukongEnv
from nn_model import DQNEfficientNet
from ddqn_agent import DDQNAgent
import os
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('toruch device:',device)
env = BlackMythWukongEnv()
policy_net = DQNEfficientNet(env.action_space.n).to(device)
target_net = DQNEfficientNet(env.action_space.n).to(device)
loss_fn = F.mse_loss
# Assuming `model` is an instance of DQNEfficientNet and learning_rate is defined
learning_rate = 1e-4

# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam([
    {'params': policy_net.features.parameters(), 'lr': 1e-5},  # Low LR for pre-trained backbone
    {'params': policy_net.fc.parameters(), 'lr': 1e-3}  # Higher LR for classifier
],lr=0.001)
# for param in model.features.parameters():
#     param.requires_grad = False  # Freeze pre-trained layers
agent = DDQNAgent(env.action_space, 
                  policy_net=policy_net,
                  target_net=target_net,
                  optimizer=optimizer, loss_fn=loss_fn, device=device)

episodes = int(1e9)
batch_size = 32
losses = []
def obs2stateTensor(obs, show=False):
    obs = obs[:, 420:1500] # choose area for model input
    state = cv2.resize(obs, (224, 224))
    if(show): 
        cv2.imshow('model input', cv2.cvtColor(state, cv2.COLOR_RGB2BGR))
    state = torch.FloatTensor(state)
    state = state.permute(2, 0, 1)  # (H, W, C) -> C, H, W
    return state

t1 = time.time()
for episode in range(episodes):
    state = env.reset()
    state = obs2stateTensor(state, show=True)

    done = False
    total_reward = 0
    
    frame_count = 0
    episode_steps = 0
    while not done:
        action = agent.select_action(state)
        obs_img, reward, done, info = env.step(action)
        if(not info['pasued']):
            # not pased
            total_reward += reward
            # print('reward',reward)
            next_state = obs2stateTensor(obs_img, show=True)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            # agent.replay(batch_size)
            # agent.update_epsilon()
            # if(frame_count%20==0):
            #     agent.update_target_net()
            total_reward += reward
            episode_steps += 1
        if time.time() - t1 > 1:
            FPS = frame_count / (time.time() - t1)
            frame_count = 0
            t1 = time.time()
            img = obs_img 
            env.draw_areas(img)
            print(f'[{"Paused" if info["pasued"] else "Trainning"}] Ep: {episode}, FPS: {FPS:.2f}, total reward: {total_reward:.2f}, \
Epsilon: {agent.epsilon:.2f}, \
wukong (H,M,S,F): {(info["wukong_health"], info["wukong_mana"], info["wukong_stamina"], info["wukong_focus"])}, boss: {info["boss_health"]}')
            cv2.imshow('Game Analysis', cv2.cvtColor(cv2.resize(img, (img.shape[1]//2, img.shape[0]//2)), cv2.COLOR_RGB2BGR))
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        frame_count += 1
        time.sleep(0.005)
    
    epLosses = []
    if(len(agent.memory) < batch_size):
        print(f"Episode Done: {episode}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
        continue
    else:
        print('episode done, Training ...')
        
    # Update network after episode
    for i in range(episode_steps):
        loss = agent.replay(batch_size)
        agent.update_epsilon()
        print(f'{i/episode_steps*100:.1f}% done, {i} of {episode_steps}, loss: {loss:.4f}', end='\r')
        epLosses.append(loss)
        # if i%20==0:
        #     agent.update_target_net()
    # Update target network after every episode
    agent.update_target_net()
    print(f"Episode Done: {episode}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}, loss: {np.mean(epLosses):.4f}")
    # save target model
    folder = f'./ddqn/wukong_off_policy/'
    os.makedirs(folder, exist_ok=True)
    torch.save(agent.target_net.state_dict(), f'{folder}/model_{episode}.pth')
    # draw graph
    losses.extend(epLosses)
    plt.plot(losses)
    plt.yscale('log')
    plt.draw()
    plt.pause(0.01)

cv2.destroyAllWindows()
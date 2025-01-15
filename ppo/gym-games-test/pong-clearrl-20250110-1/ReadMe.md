# Teseted model and hyperparameter

- seed of random affect training process. Sometime, it can be failed training.
  - e.g runs\PongNoFrameskip-v4__ppo_atari__1__1736710905 and runs\PongNoFrameskip-v4__ppo_atari__2__1736714626 has the same model. The only difference is seed. The former one with seed 1 and later is seed 2. former one is failed to get a working model, while seed 2 converged not bad. 

- weight initialization method. 
  - Default torch initalization works, it genally slower. 
  - torch.nn.init.orthogonal_(layer.weight, np.sqrt(2)) for CNN net, could speed up converge speed 

- model network structure

# Appendix
## Kaboom-v5, dqn vs PPO
![Kaboom-v5](./fig/dqn-training-labels-2.png)

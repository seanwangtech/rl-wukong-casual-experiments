# fixed Highligh issues

- **Major**: In the ppo_agent.py, **fixed the way to calculate "Returns"**. returns refer to the cumulative rewards collected by an agent from a given state onward in an episode. Returns can be used to estimate state values with Monte Carlo (MC) algoirthm. This is very impotant. 
```python
# below method work greatly
returns = advantages + values

# not sure why below doesn't work. 
returns = []
G = 0
for reward, done in zip(reversed(rewards), reversed(next_dones)):
    G = reward + self.gamma * G * (1 - done)
    returns.append(G)
returns.reverse()
```
- **Major**: In the ppo_agent.py, fixed normalization of advantages. Normalize the advantage in every mini-batch instead of whole episode.
- *Minor*: In the ppo_agent.py, duplicate smaple to integer number of mini-batch size to ensure the same size for each mini-batch. Slightly increase model quality. 
- *Minor*: normalize the observiation/model input. initially, I forget to divide the gym env observation by 255.0 to normalize data between 0 to 1. Therefore, the mode takes long time to converge. After normalizaion, it converage much fast. Slightly increase model quality. 
- **Major**: entropy loss coefficient, revised from 0.1 to 0.01. Regarding Entropy Regularization, Paper suggest coefficient 0 to 0.01. Original setting is too large which result in unstable and bad result. cite from paper: The entropy coefficient is multiplied by the maximum possible entropy and added to loss. This helps prevent premature convergence of one action probability dominating the policy and preventing exploration. 

## plots


### Highligh issues repair plot 
The notes for below plots, in the order of time.  
![trains history](./fig/train-result-and-algoirthm-tune.png)
- Before repair issues
- Nomalize the advance in mini-batch instead of whole episode and fixed the way to calculate returns
- change (batch_size:256, update_epochs:10) to (batch_size:32, update_epochs:4)
- Initialize model weights by using torch.nn.init.orthogonal_()
- Normalize the input by divide x by 255.0
- revised entropy loss coefficient from 0.1 to 0.01
- Increase the batch_size from 32 to 128 to stablize the output. 

### entropy loss coefficient
![alt text](fig/train-reults-entropy.png)
- revised entropy coefficient 0.1
- revised entropy coefficient 0.01 - first run 
- revised entropy coefficient 0.01 - second run
- revised entropy coefficient 0.01 - third run

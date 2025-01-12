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
- *Minor*: In the ppo_agent.py, duplicate smaple to integer number of mini-batch size to ensure the same size for each mini-batch. 
- **Major**: normalize the observiation/model input. initially, I forget to divide the gym env observation by 255.0 to normalize data between 0 to 1. Therefore, the model doesn't work well, can only achieve Pong score around +14 and it take long time. After normalizaion, it converage much fast with much better result > +18 ??? to be confirmed. 

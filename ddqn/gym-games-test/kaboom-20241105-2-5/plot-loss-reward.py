import matplotlib.pyplot as plt
import pandas as pd
import os
plt.figure(figsize=(10, 5))
pltsubs = (plt.subplot(1, 2, 1), plt.subplot(1, 2, 2))
pltsubs[0].set_yscale('log')
df = pd.read_csv(f'{os.path.dirname(__file__)}/trains/log_1220.csv')
losses = df['loss'].tolist()
total_rewards = df['reward'].tolist()
pltsubs[0].plot(losses)
pltsubs[1].plot(total_rewards)
plt.show()
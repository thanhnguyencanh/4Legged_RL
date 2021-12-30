# Reinforcement Learning for 4 Legged Robot


## Getting Started

These instructions will demonstrate how to setup a conda environment with all requirements for the project setup.

### Installing

```
conda env create -n rl_dev python=3.6

conda activate rl_dev

git clone https://github.com/NguyenCanhThanh/4Legged_RL.git

cd 4Legged_RL

python setup.py install 

jupyter notebook
```

### Results

The notebook uses the same hyperparameters and architecture described in the paper. The agent is trained for 5 million timesteps. The agent converged on a successfull policy after 400k timesteps. The results below show the agents avg score over the previous 100 episodes.

As you can see, the agent learned rapidly and then briefly fell into a local optima. However, the agent was able to quickly recover itself. I believe with hyperparameter tuning and a proper sample of trained agents, the results could still improve. 

<img src="Output/PPO.png">
<img src="Output/sac.png">
<img src="Output/TD3.png">


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dHWlM7BWYqQ4Yc_LuQYij5zWhSapzxAD?usp=sharing)

An implementation of the TD3 algorithm trained on the Roboschool HalfCheetah environment using pytorch. The code here is based on the work of the original authors of the TD3 algorithm found [here](https://github.com/sfujim/TD3). 

## Acknowledgments

* OpenAI [Spinning Up](https://github.com/openai/spinningup)
* OpenAI [Baselines](https://github.com/openai/baselines)

<img width="160px" height="22px" href="https://github.com/pytorch/pytorch" src="https://pp.userapi.com/c847120/v847120960/82b4/xGBK9pXAkw8.jpg">


PyTorch code of: proximal policy optimization / ddpg / twin dueling ddpg / soft actor critic


  1.  Proximal Policy Optimization Algorithms 
  - [ppo.ipynb](https://github.com/NguyenCanhThanh/4Legged_RL/blob/main/ppo.ipynb)
  - [PPO Paper](https://arxiv.org/abs/1707.06347)
  - [OpenAI blog](https://blog.openai.com/openai-baselines-ppo/)
  2.  Continuous control with deep reinforcement learning
  - [ddpg.ipynb](https://github.com/NguyenCanhThanh/4Legged_RL/blob/main/ddpg.ipynb)
  - [DDPG Paper](https://arxiv.org/abs/1509.02971)
  3. Addressing Function Approximation Error in Actor-Critic Methods
  - [td3.ipynb](https://github.com/NguyenCanhThanh/4Legged_RL/blob/main/td3.ipynb)
  - [Twin Dueling DDPG Paper](https://arxiv.org/abs/1802.09477)
  4. Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor 
  - [soft actor-critic.ipynb](https://github.com/NguyenCanhThanh/4Legged_RL/blob/main/soft%20actor-critic.ipynb)
  - [Soft Actor-Critic Paper](https://arxiv.org/abs/1801.01290)


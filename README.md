# Reinforcement Learning for 4 Legged Robot

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dHWlM7BWYqQ4Yc_LuQYij5zWhSapzxAD?usp=sharing)

An implementation of the TD3 algorithm trained on the Roboschool HalfCheetah environment using pytorch. The code here is based on the work of the original authors of the TD3 algorithm found [here](https://github.com/sfujim/TD3). 

<img src="media/trained_agent.gif">

## Getting Started

These instructions will demonstrate how to setup a conda environment with all requirements for the project setup.

### Installing

```
conda env create -n rl_dev python=3.6

conda activate rl_dev

git clone https://github.com/djbyrne/TD3.git

cd TD3

python setup.py install 

jupyter notebook
```

### Results

The notebook uses the same hyperparameters and architecture described in the paper. The agent is trained for 5 million timesteps. The agent converged on a successfull policy after 500k timesteps. The results below show the agents avg score over the previous 100 episodes.

As you can see, the agent learned rapidly and then briefly fell into a local optima. However, the agent was able to quickly recover itself. I believe with hyperparameter tuning and a proper sample of trained agents, the results could still improve. 

<img src="media/training_results.png">


## Acknowledgments

* Scott Fujimoto [TD3](https://github.com/sfujim/TD3)
* OpenAI [Spinning Up](https://github.com/openai/spinningup)
* OpenAI [Baselines](https://github.com/openai/baselines)

<img width="160px" height="22px" href="https://github.com/pytorch/pytorch" src="https://pp.userapi.com/c847120/v847120960/82b4/xGBK9pXAkw8.jpg">


PyTorch tutorial of: actor critic / proximal policy optimization / acer / ddpg / twin dueling ddpg / soft actor critic / generative adversarial imitation learning / hindsight experience replay

The deep reinforcement learning community has made several improvements to the [policy gradient](http://rll.berkeley.edu/deeprlcourse/f17docs/lecture_4_policy_gradient.pdf) algorithms. This tutorial presents latest extensions in the following order: 

1. Advantage Actor Critic (A2C)
 - [actor-critic.ipynb](https://github.com/higgsfield/RL-Adventure-2/blob/master/1.actor-critic.ipynb)
 - [A3C Paper](https://arxiv.org/pdf/1602.01783.pdf) 
 - [OpenAI blog](https://blog.openai.com/baselines-acktr-a2c/#a2canda3c)
  2. High-Dimensional Continuous Control Using Generalized Advantage Estimation
  - [gae.ipynb](https://github.com/higgsfield/RL-Adventure-2/blob/master/2.gae.ipynb)
  - [GAE Paper](https://arxiv.org/abs/1506.02438)
  3.  Proximal Policy Optimization Algorithms 
  - [ppo.ipynb](https://github.com/higgsfield/RL-Adventure-2/blob/master/3.ppo.ipynb)
  - [PPO Paper](https://arxiv.org/abs/1707.06347)
  - [OpenAI blog](https://blog.openai.com/openai-baselines-ppo/)
  4.  Sample Efficient Actor-Critic with Experience Replay 
  - [acer.ipynb](https://github.com/higgsfield/RL-Adventure-2/blob/master/4.acer.ipynb)
  - [ACER Paper](https://arxiv.org/abs/1611.01224)
  5.  Continuous control with deep reinforcement learning
  - [ddpg.ipynb](https://github.com/higgsfield/RL-Adventure-2/blob/master/5.ddpg.ipynb)
  - [DDPG Paper](https://arxiv.org/abs/1509.02971)
  6. Addressing Function Approximation Error in Actor-Critic Methods
  - [td3.ipynb](https://github.com/higgsfield/RL-Adventure-2/blob/master/6.td3.ipynb)
  - [Twin Dueling DDPG Paper](https://arxiv.org/abs/1802.09477)
  7. Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor 
  - [soft actor-critic.ipynb](https://github.com/higgsfield/RL-Adventure-2/blob/master/7.soft%20actor-critic.ipynb)
  - [Soft Actor-Critic Paper](https://arxiv.org/abs/1801.01290)
  8.  Generative Adversarial Imitation Learning 
  - [gail.ipynb](https://github.com/higgsfield/RL-Adventure-2/blob/master/8.gail.ipynb)
  - [GAIL Paper](https://arxiv.org/abs/1606.03476)
  9.  Hindsight Experience Replay
  - [her.ipynb](https://github.com/higgsfield/RL-Adventure-2/blob/master/9.her.ipynb)
  - [HER Paper](https://arxiv.org/abs/1707.01495)
  - [OpenAI Blog](https://blog.openai.com/ingredients-for-robotics-research/#understandingher)

# If you get stuck… 
- Remember you are not stuck unless you have spent more than a week on a single algorithm. It is perfectly normal if you do not have all the required knowledge of mathematics and CS.
- Carefully go through the paper. Try to see what is the problem the authors are solving. Understand a high-level idea of the approach, then read the code (skipping the proofs), and after go over the mathematical details and proofs.

# RL Algorithms
Deep Q Learning tutorial: [DQN Adventure: from Zero to State of the Art](https://github.com/higgsfield/RL-Adventure)
[![N|Solid](https://planspace.org/20170830-berkeley_deep_rl_bootcamp/img/annotated.jpg)]()
Awesome RL libs: rlkit [@vitchyr](https://github.com/vitchyr), pytorch-a2c-ppo-acktr [@ikostrikov](https://github.com/ikostrikov),
ACER [@Kaixhin](https://github.com/Kaixhin)

# Best RL courses
- Berkeley deep RL [link](http://rll.berkeley.edu/deeprlcourse/)
- Deep RL Bootcamp [link](https://sites.google.com/view/deep-rl-bootcamp/lectures)
- David Silver's course [link](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)
- Practical RL [link](https://github.com/yandexdataschool/Practical_RL)


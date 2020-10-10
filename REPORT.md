[//]: # (Image References)


[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42386929-76f671f0-8106-11e8-9376-f17da2ae852e.png "Kernel"
[image3]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"

# Goal

The project aim is to train two agents that control rackets and bounce a ball over the a net. If an agent hits the ball over the net,
it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.
The goal of each agent is to keep the ball in play.


![Trained Agent][image1]

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives
its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5
(over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

# Approach

I started with the DDPG [2] algorithm that I developed for [Continuous Control](https://github.com/miharothl/DRLND-Collaboration-And-Competition)[1]. I adjusted the configuration
to use 2 agents and was able to solve the environment by achieving a score 0.5+ over 100 consecutive episodes.

I tired to improve the learning process by implementing:
- PER [3]
- MADDPG [4]

The steps that I followed to solve this environment:

1. Evaluate the state and action space of the environment
2. Establish a baseline using a random action policy
3. Implement the learning algorithm
4. Run experiments and select the best agent

## 1. Evaluate State & Action Space of the Environment

The action-space has 2 continuous dimensions corresponding to the, corresponding to movement toward (or away from) the net, and jumping. 
The state-space is continuous; it has 8 dimensions corresponding to the position and velocity of the ball and racket. Each agent receives
its own, local observation.

## 2. Establish Baseline Using Random Action Policy

Before starting the deep reinforcement learning process, its good to understand the environment. Controlling the 
rackets with an agent where actions have randomly selected achieve scores averaging 0.01 over 100 consecutive episodes.
 
## 3. Implement Learning Algorithm

The
[agent](https://github.com/miharothl/DRLND-Collaboration-And-Competition/blob/master/drl/agent/ddpg_agent.py)
and 
[environment](https://github.com/miharothl/DRLND-Collaboration-And-Competition/blob/master/drl/env/unity_multiple_env.py)
are created according to the provided
[configuration](https://github.com/miharothl/DRLND-Collaboration-And-Competition/blob/master/drl/experiment/configuration.py)
.
[Recorder](https://github.com/miharothl/DRLND-Collaboration-And-Competition/blob/master/drl/experiment/recorder.py)
records the experiment and store the results for later
[analysis](https://github.com/miharothl/DRLND-Collaboration-And-Competition/blob/master/rlab-analysis.ipynb)
.

The agent interacts with the environment in the
[training loop](https://github.com/miharothl/DRLND-Collaboration-And-Competition/blob/master/drl/experiment/train/master_trainer.py)
.
In the exploration phase (higher *Epsilon*) of the training
agent's actions are mostly random, created using 
[Ornstein-Uhlenbeck noise generator](https://github.com/miharothl/DRLND-Collaboration-And-Competition/blob/master/drl/agent/tools/ou_noise.py)
. Actions, environment states, dones, and rewards tuples, are stored in the experience
replay buffer. The *Buffer Size* parameter determines the size of the buffer.

DDPG [3] is using 
[actor and critic](https://github.com/miharothl/DRLND-Collaboration-And-Competition/blob/master/drl/model/ddpg_model.py)
neural networks. Both have current, and target model with identical architecture used to stabilize the DDPG learning process.
During the learning process, weights of the target network are fixed (or updated more slowly based on parameter *Tau*).

The loss function is defined as the mean square error of critic's temporal difference error, the difference between the expected
and estimated q values. Adam optimizer minimizes the loss function performing the gradient descent and backpropagation algorithm
using the specified *Learning Rate*.

Learning is performed *Num Updates* times on every *Update Every* steps, when *Batch Size* of actions, states, dones and rewards tuples are
sampled from the
[replay buffer](https://github.com/miharothl/DRLND-Collaboration-And-Competition/blob/master/drl/agents/replay_buffer.py)
either randomly or in case of prioritized experience replay, based on their importance,
determined by the temporal difference error. Prioritized experience replay requires
[segment trees](https://github.com/miharothl/DRLND-Collaboration-And-Competition/blob/master/drl/agents/segment_tree.py)
.

During the exploitation phase of the training (lower *Epsilon*) the noise added to the actions is proportionally scaled down (*epsilon end*)
and mostly based on the estimated policies calculated by the current actor neural network.

## 4. Run Experiments and Select Best Agent

[Training](https://github.com/miharothl/DRLND-Collaboration-And-Competition/blob/master/rlab-continous-control.ipynb)
is done using the epochs, consisting of training episodes where epsilon greedy agent is used,
and validation episodes using only actions predicted by the trained agent.
 
I used the following training hyperparameters:

|Hyper Parameter            |Value                 |
|:---                       |:---                  |
|Max Steps                  |300000 (300 episodes) |
|Max Episode Steps          |1000                  |
|Evaluation Frequency       |10000  (10 episodes)  |
|Evaluation Steps           |2000   (2 episodes)   |
|Epsilon Start              |1.                    |
|Epsilon End                |0.1                   |
|Epsilon Decay              |0.97                  |
|Batch Size                 |128                   |
|Update Every               |10                    |
|Num Updates                |20                    |
|Learning Rate Actor        |0.0001                |
|Learning Rate Critic       |0.0003                |
|Tau                        |0.001                 |
|Gamma                      |0.99                  |
|Actor Hidden Layers Units  |[256, 128]            |
|Critic Hidden Layers Units |[256, 128]            |
|Buffer Size                |100000                |
|Use Prioritized Replay     | False                |

The first version of an agent that can solve the environment with scores 30+ is obtained in 1st epoch after 19 training episodes. 

![Training Score][image4]
![Training Epsilon][image5]

```
2020-09-24 18:29:49,164 - drl - EPISODE - Train. - {'step': 18000, 'episode': 17, 'epoch': 1, 'epoch_step': 8000, 'epoch_episode': 8, 'episode_step': 999, 'score': '24.504', 'eps': '0.596', 'elapsed': '87s'}
2020-09-24 18:31:16,644 - drl - EPISODE - Train. - {'step': 19000, 'episode': 18, 'epoch': 1, 'epoch_step': 9000, 'epoch_episode': 9, 'episode_step': 999, 'score': '24.781', 'eps': '0.578', 'elapsed': '87s'}
2020-09-24 18:32:44,233 - drl - EPISODE - Train. - {'step': 20000, 'episode': 19, 'epoch': 1, 'epoch_step': 10000, 'epoch_episode': 10, 'episode_step': 999, 'score': '27.614', 'eps': '0.561', 'elapsed': '88s'}
2020-09-24 18:34:03,759 - drl - EPISODE - Validate. - {'epoch': 1, 'epoch_step': 1000, 'epoch_episode': 1, 'episode_step': 999, 'score': '30.364', 'eps': '0.544', 'elapsed': '80s'}
2020-09-24 18:35:23,298 - drl - EPISODE - Validate. - {'epoch': 1, 'epoch_step': 2000, 'epoch_episode': 2, 'episode_step': 999, 'score': '30.859', 'eps': '0.544', 'elapsed': '80s'}
```

The best agent is trained in epoch 29 after playing 299 episodes and can achieve a score **38.84** over 100 consecutive episodes using multiple 20 agents.

```
2020-09-25 08:40:34,511 - drl - EPISODE - Play. - {'episode': 97, 'score': '39.093', 'elapsed': '79.617s'}
2020-09-25 08:41:54,146 - drl - EPISODE - Play. - {'episode': 98, 'score': '38.862', 'elapsed': '79.619s'}
2020-09-25 08:43:13,783 - drl - EPISODE - Play. - {'episode': 99, 'score': '38.480', 'elapsed': '79.619s'}

Average score over 100 episodes is 38.83892413188238
```

# Future Work

Deep reinforcement learning is a fascinating and exciting topic. I'll continue to improve my reinforcement learning
laboratory by applying
 * distributed algorithms like PPO [4], A3C [5] or D4PG [6] and by
 * solving other attractive environments requiring continuous control like the Crawler.

![Crawler][image3]

# References
  - [1] [Udacity](https://github.com/udacity/deep-reinforcement-learning)
  - [2] [Open AI Baselines](https://github.com/openai/baselines)
  - [3] [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)
  - [4] [Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347.pdf) 
  - [5] [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1602.01783.pdf)
  - [6] [Distributed Distributional Deterministic Policy Gradients](https://openreview.net/pdf?id=SyZipzbCb)


A project aiming to play chrome dino with Deep Q-learning.
Online game website: (https://chromedino.com)

![chrome dino](https://storage.googleapis.com/gweb-uniblog-publish-prod/original_images/Dino_non-birthday_version.gif)

# MDP 
A Markov decision process is a discrete time stochastic control process. At each time step, the process is in some state s, and the
decision maker may choose any action a that is available in state s. The process responds at the next time step by moving into 
a new state s′,and giving the decision maker a corresponding reward Ra(s,s′). 
In our case, the decision maker is the little dino, while the screenshots which the dino can see are states.

# DQN
Our goal is to find the optimal policy, which is equavaliant to finding to optimal all action-values functions. Becasue overall we aim to
maximise the expected return from the very first time step onwards. With a MDP model (the rewards and the transition probability matrix are
avaliable), we can achieve with policy or value iteration. However, for this game, the MDP model is unknown, but we can approximate the
action-value function by monte carlo method or temporal difference learning. This states of the game is huge, hence we apply the DQN with 
experience replay to find the optimal action-value functions.

# Why using experience replay?
Make data close to i.i.d; better convergence behavior; more efficient use of data.

# Triky part of this task
Defining the reward function is hard.
Tuning of hyper-parameters.
The game speed is getting faster when you play.
The blackground will shift between day and night after 700 points.

# Best performance
The highest score is around 1000.

# Reproduce
It maybe a bit hard to reproduce my results because I am iteracting with the game by purely looking at the screenshots, so you may need to 
adjust the bounding bboxes for detecting start, end of the game.

# Video
For demo, please go to https://www.youtube.com/watch?v=nC1GX7X_aHA&feature=youtu.be

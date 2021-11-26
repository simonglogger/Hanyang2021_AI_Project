# Hanyang2021_AI_Project

## 1. Introduction
The main target of this project is to train an agent how to play the game flappy bird using neural networks and reinforcement learning (Deep Q Learning). 
The basic version of the game itsself can be downloaded from grantjenks. Nevertheless, the game is adapted massively for the present project. 
The goal of flappy bird is to pass in between two incoming balls in order to survive and increase the score. Since flappy bird moves down constantly, the only available action a player can choose is a vertical jump. As soon as flappy bird collides with the walls or the balls, the game is over. The score is defined by the number of ball pairs flappy bird passes.
In this project, we are going to train a feedforward neural network using the game state information as input, 
the action (jump (1) or don't jump, i.e. do nothing (0)) as output and a reward function. For training the neural network, 
we are going to use backpropagation. The agent is supposed to improve and increase the score troughout training.

<img src="Result_Training_Simple_Game.png">

Sources: 
Game: http://www.grantjenks.com/docs/freegames/flappy.html  
DQN: https://github.com/the-deep-learners/TensorFlow-LiveLessons/blob/master/notebooks/cartpole_dqn.ipynb


## 2. Datsaets/Game

## 3. Reinrorcement Learning/Deep Q Learning

State, Action, Reward, Greedy Policy, Epsilon, Epsilon Decay, Replay Memory, Batch Size, Bellman Equation, Q-Learning, Difference to Deep Q Learning, 

L(θ)=E(s,a,r,s′)∼U(D)[(r+γmaxa′Q(s′,a′;θ−)−Q(s,a;θ))2]

Link: https://ai.stackexchange.com/questions/25086/how-is-the-dqn-loss-derived-from-or-theoretically-motivated-by-the-bellman-equ

## Implementation in Python

## 4. Evaluation and Analysis

## 5. Related Work

## 6. Conclusion

## Titel: Training an agent to play flappy bird using reinforcement learning.



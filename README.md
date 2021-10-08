# Hanyang2021_AI_Project

Titel: Training an agent to play flappy bird using reinforcement learning.

The main target of this project is to train an agent how to play the game flappy bird using neural networks and reinforcement learning. 
The game itsself can be downloaded from github. The goal of flappy bird is to avoid incoming balls in order to survive and increase the score. 
As flappy bird moves down constantly, the only available action a player can use is a vertical jump. 
As soon as flappy bird collides with the walls or the balls, the game is over. The score is defined by the time flappy bird survives.

In this project, we are going to train a feedforward neural network using the game state information as input, 
the action (jump (1) or don't jump (0)) as output and a reward function that is supposed to be maximized. For training, 
we are going to use backpropagation. The agent is supposed to improve and increase the score troughout training.

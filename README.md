# Hanyang2021_AI_Project

## 1. Introduction
The main target of this project is to train an agent to play the game flappy bird using neural networks and reinforcement learning (deep q learning).   
  
Flappy Bird tries to pass in between two incoming balls in order to survive and increase the score. Since flappy bird moves down constantly, the only available action an agent can choose is a vertical jump. As soon as flappy bird collides with the walls or the balls, the game is over. The score is defined by the number of ball pairs flappy bird passes.  
  
In this project, we are going to train a feedforward neural network using the game state information as input, 
the action (jump (1) or don't jump, i.e. do nothing (0)) as output and a reward function. For training the neural network, 
we are going to use backpropagation. The agent is supposed to improve and increase the score troughout training.  

<img src="Result_Training_Simple_Game.png">

Sources: 
Game: http://www.grantjenks.com/docs/freegames/flappy.html  
DQN: https://github.com/the-deep-learners/TensorFlow-LiveLessons/blob/master/notebooks/cartpole_dqn.ipynb


## 2. Datsaets/Game

## 3. Reinrorcement Learning/Deep Q Learning

In order to train our agent to play Flappy Bird by itself, we chose to use Reinforcement Learning, and more precisely Deep Q Learning. Reinforcement Learning is one of the three main fields of Machine Learning, alongside supervised and unsupervised Learning.

The working principle of Reinforcement Learning is that there is an agent leaving in an environment.The environment gives the agent a state, in return the agent takes an action, and then the environment provides a numerical reward in response to this action to the agent, as well as its next state. This process is going to be repeated ate every step and the goal is to learn how to take the best actions in order to maximize the reward received.

A Markov Decision Problem is the mathematical formulation of the Reinforcement Learning problem. It satisfies the Markov proprety, which is that the current state completely characterises the state of the world. An MDP is characterized by
- a set of possible states
- a set of possible actions
- distribution of reward
- transition probability over next state 
- a discount factor (how much we value rewards coming on soon vs later on)

A policy pi is a function from the S (the set of states) to A (the state of action) that specifies what action has to be taken at every state. The goal is going to find the optimal policy pi* that maximizes cumulative discounted reward. In order to handle the randomness of the interaction between the environment and the agent, we maximize the expected sum of rewards.

How can we tell how good a state is? We use the value function at state s, which is the expected cumulative reward when using the policy pi.
How good is a state-action pair? To quantify that, we use the Q-value function at state s and action a, which is the expected cumulative reward from taking action a in state s and then following the policy.
The optimal Q-value function that we can get is Q*, the maximum expected cumulative reward achievable from a given state-action pair.
Q* satisfies the Bellman equation (given any state-action pair, the value of this pair is going to be the value of the reward that we are going to get plus the value of whatever state we are going to get). 
Our optimal policy is going to consist in taking the best action at any state as specified by Q*.
One way to solve it is with a value iteration algorithm, where the Bellman equation is being used as an iterative update
Problem : not scalable, must compute Q(s,a) for every state-pair action. We can use instead a neural network as a function approximator
When the function approximator is a deep neural network -> Deep Q Learning
We want to find a function that satisfies the Bellman equation, we want to get as close to the expected reward as possible (iteratively)
The loss function of the gradient update by backward pass is going to be

Experience replay : learning from batches of consecutive samples is bad because consecutive samples are strongly correlated -> unefficient learning
Keep replay memory table of transitions as episodes are played, an train Q-network on random minibatches of transitions from the replay memory

Initialize replay memory D to size N  
Initialize action-value function Q with random weights  
for episode = 1, M do  
    Initialize state s_1  
    for t = 1, T do  
        With probability ϵ select random action a_t  
        otherwise select a_t=max_a  Q(s_t,a; θ_i)  
        Execute action a_t in emulator and observe r_t and s_(t+1)  
        Store transition (s_t,a_t,r_t,s_(t+1)) in D  
        Sample a minibatch of transitions (s_j,a_j,r_j,s_(j+1)) from D  
        Set y_j:=  
            r_j for terminal s_(j+1)  
            r_j+γ* max_(a^' )  Q(s_(j+1),a'; θ_i) for non-terminal s_(j+1)  
        Perform a gradient step on (y_j-Q(s_j,a_j; θ_i))^2 with respect to θ  
    end for  
end for  


State, Action, Reward, Greedy Policy, Epsilon, Epsilon Decay, Replay Memory, Batch Size, Bellman Equation, Q-Learning, Difference to Deep Q Learning, 

L(θ)=E(s,a,r,s′)∼U(D)[(r+γmaxa′Q(s′,a′;θ−)−Q(s,a;θ))2]

Link: https://ai.stackexchange.com/questions/25086/how-is-the-dqn-loss-derived-from-or-theoretically-motivated-by-the-bellman-equ

## 4. Implementation in Python
In this project, three different python scripts are used for the training framework. The first script (Game_Functions) contains the game itsself. It's recieves an action and basically computes the new state and the reward. Another script (Agent) contains the class of the agent. In general, it gets the current game state and attempts to predict the most suitable action. Moreover, it stores the data for the replay memory. The thrid script (Main_Control) calls the other two scripts alternately and contains the training of the agent.
  
The algorith basically looks like this:  
  
  import packages
  initialize game  
  initialize agent  
  for i in range(n_episodes)  
      get initial state  

      while bird_dead == False:
        action = agent(state) || f(random)
        replay_memory_data (e.g. reward) = game(action)

        if (score > max_steps) & (epsilon < 0.6)
          save nn
          learning_rate *= 0.1
          max_steps += XX

      training_data = random.sample(memory)
      prediction = agent(training_data)
      loss = lossf(target, prediction) 

      agent.optimizer.zero_grad() 
      loss.backward() 
      agent.optimizer.step()
    
 
      


## 5. Evaluation and Analysis

## 6. Conclusion



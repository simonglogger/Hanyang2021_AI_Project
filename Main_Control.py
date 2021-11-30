import turtle as tl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt
from freegames import vector
from Game_Functions import main
from Agent import Agent

input_size = 6
output_size = 2
batch_size = 32
n_episodes = 300
max_steps = 2000
step_count_append = np.array([])
epsilon_append = np.array([])
score_append = np.array([])

agent = Agent(input_size, output_size)

for i in range (n_episodes):
    print("Current Episode: ", i)
    
    bird = vector(0, 0)                    
    balls_low = [vector(199, -100)]         
    balls_high = [vector(199, 100)]       
    bird_dead = False
    step_count = 1
    score = 0
    
    #tl.setup(420, 420)                     
    #tl.hideturtle()
    #tl.up()
    #tl.tracer(False)
    
    while bird_dead == False:
        if step_count == 1:
            state = [0.5, 0.0, 1.0, 1.0, 0.24874372, 0.75125628] 
            state = np.reshape(state, [1, input_size])
        
        #Play and remember
        action = agent.act(state) 
        state, action, reward, next_state, bird_dead, score = main(bird, balls_low, balls_high, action, step_count, score)
        agent.remember(state, action, reward, next_state, bird_dead)  
               
        if bird_dead == 1: 
            break 
        
        if step_count >= max_steps:
            torch.save(agent.state_dict(), 'agent_nn')
            agent.learning_rate *= 0.1
            max_steps += 500
            break
        
        step_count += 1
     
    if max_steps > 5000:
        break
    
    if len(agent.memory) > batch_size:
        model_output, target_f = agent.replay(batch_size)
                
        optim_agent = optim.Adam(agent.parameters(), lr = agent.learning_rate)
        agent.optimizer = optim_agent
                    
        loss_fcn = nn.MSELoss()
        loss = loss_fcn(model_output, target_f)
                
        agent.optimizer.zero_grad() 
        loss.backward() 
        agent.optimizer.step()
        
    if agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay 
     
    print("Epsilon: ", agent.epsilon)
    print("Score: ", score)
    print("########################")
    step_count_append = np.append(step_count_append, step_count)
    epsilon_append = np.append(epsilon_append, agent.epsilon)
    score_append = np.append(score_append, score)
    
    agent.memory.clear()

#Create x-vector for plot
length = len(score_append)
x_values = np.array([])
for i in range(length):
    x_values = np.append(x_values, i + 1)

#Result Plot Score
fig1, ax1 = plt.subplots()
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Score')
ax1.scatter(x_values, score_append, color='blue')
ax1.tick_params(axis='y', labelcolor = 'blue')
ax2 = ax1.twinx()  
ax2.set_ylabel('Epsilon')  
ax2.plot(x_values, epsilon_append, color='red')
ax2.set_ylim(0,1)
ax2.tick_params(axis='y', labelcolor = 'red')
fig1.tight_layout()
plt.show()
         
 
        


import turtle as tl
import torch.nn as nn
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt
from freegames import vector
from Game_Functions import main
from Agent import Agent

input_size = 5
output_size = 2
batch_size = 32
n_episodes = 500
step_count_append = np.array([])

agent = Agent(input_size, output_size)

for i in range (n_episodes):
    print("Current Episode: ", i)
    
    bird = vector(0, 0)                    #Initiale Position Vogel
    balls_low = [vector(199, -100)]        #Leeres array für alle Raben (Positionen)
    balls_high = [vector(199, 100)]        #Leeres array für alle Raben (Positionen)
    bird_dead = False
    step_count = 1
    
    tl.setup(420, 420)      #Spielfeld Definition (Breite, Höhe, Pos. Spielfeld X, Pos. Spielfeld Y)
    tl.hideturtle()
    tl.up()
    tl.tracer(False)
    
    while bird_dead == False:
        if step_count < 2:
            state = [0.5, 1.0, 1.0, 0.24874372, 0.75125628] 
            state = np.reshape(state, [1, input_size])
        
        #Play and remember
        action = agent.act(state) 
        state, action, reward, next_state, bird_dead = main(bird, balls_low, balls_high, action, step_count)
        agent.remember(state, action, reward, next_state, bird_dead)  
               
        if bird_dead == 1: 
            break 

        step_count += 1
        
    #Train with last game data
    print("Step Count: ", step_count)
    step_count_append = np.append(step_count_append, step_count)
    
    if len(agent.memory) > batch_size:
        
        model_output, target_f = agent.replay(batch_size)
        
        optim_agent = optim.Adam(agent.parameters(), lr = agent.learning_rate)
        agent.optimizer = optim_agent
            
        loss_fcn = nn.MSELoss()
        loss = loss_fcn(model_output, target_f)
            
        agent.optimizer.zero_grad() 
        loss.backward() 
        agent.optimizer.step()
        
        agent.memory.clear()

length = len(step_count_append)
x_values = np.array([])
for i in range(length):
    x_values = np.append(x_values, i + 1)
'''
plt.scatter(x_values, step_count_append)
plt.xlabel('Episodes')
plt.ylabel('Steps')
plt.show()
'''            
 
        


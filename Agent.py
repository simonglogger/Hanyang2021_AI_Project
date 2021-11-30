import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn  


class Agent(nn.Module):
    def __init__(self, input_size, output_size):
        super(Agent, self).__init__() 
        
        self.input_size = input_size
        self.output_size = output_size
        self.memory = deque(maxlen=2000) 
        self.gamma = 0.95 
        
        #Training = 0, Testing = 1 !!!
        use_case = 1
        if use_case == 0:
            self.epsilon = 1 
        elif use_case == 1:
            self.epsilon = 0
        else:
            print("Wrong setting!!")
            
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.001 
        self.learning_rate = 0.01 
        
        self.model = nn.Sequential( 
            nn.Linear(input_size, 24), 
            nn.ReLU(), 
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, output_size)) 
    
    def forward(self, data): 
        return self.model(data)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) 

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            if random.randrange(4) == 0:
                y = 1
            else:
                y = 0
            return y  
        
        state = torch.from_numpy(state).float()
        act_values = self.model.forward(state) 
        act_values = act_values.detach().numpy()
        #print("Agent Action ", act_values)
        #print("Action: ", np.argmax(act_values))
        return np.argmax(act_values) 

    def replay(self, batch_size): 
        minibatch = random.sample(self.memory, batch_size) 
        
        for state, action, reward, next_state, done in minibatch: 
            
            state = torch.from_numpy(state).float()
            next_state = torch.from_numpy(next_state).float()
            
            target = reward  
            
            if not done:
                target = (reward + self.gamma * np.amax(self.model.forward(next_state).detach().numpy())) 
            target_f = self.model.forward(state)
            target_f[0][action] = target
            
            model_output = self.model.forward(state)
     
        return model_output, target_f
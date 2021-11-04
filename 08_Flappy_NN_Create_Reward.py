import random as rd
import turtle as tl
import torch 
import torch.nn as nn 
import torch.optim as optim
import numpy as np

from freegames import vector
from time import sleep


########################################### 
class Agent(nn.Module): 
    def __init__(self): 
        super(Agent, self).__init__() 
        
        self.model = nn.Sequential( 
            nn.Linear(80003, 100), 
            nn.Sigmoid(), 
            nn.Linear(100, 1), 
            nn.Sigmoid(),  
        ) 
        
    def forward(self, data): 
        return self.model(data)
 
###########################################
def scale_input(pos_bird):
    ##Position Bird
    pos_min = -200
    pos_zero = 0
    pos_max =  200
    
    pos_x_norm = (pos_bird[0] - pos_min) / (pos_max - pos_min)
    pos_y_norm = (pos_bird[1] - pos_min) / (pos_max - pos_min)
    
    pos_bird_norm = np.array([pos_x_norm, pos_y_norm])
    
    ##Distace Walls
    y_abs = abs(pos_bird[1])
    
    dis_wall_norm = (y_abs - pos_zero) / (pos_max - pos_zero)
    
    return (pos_bird_norm, dis_wall_norm)

###########################################
def binary(val):
    if val < 0.5:
        out = 0
    elif val >= 0.5:
        out = 1
    
    return out
            
###########################################
def tap():
    "Move bird up in response to screen tap."
    up = vector(0, 30) 
    bird.move(up)

###########################################
def inside(point):
    "Return True if point on screen."
    return -200 < point.x < 200 and -200 < point.y < 200 

###########################################
def draw(bird_dead):
    "Draw screen objects."
    tl.clear() 
    
    tl.goto(bird.x, bird.y) 

    if bird_dead:
        tl.dot(10, 'red')
    else:
        tl.dot(10, 'green')

    for ball in balls:
        tl.goto(ball.x, ball.y)
        tl.dot(20, 'black')

    tl.update()  
    
###########################################
class Training_Framework(): 
    def __init__(self): 
        super().__init__() 
        
        #Initialization 
        self.agent = Agent()
        
        #Optimizer
        optim_agent = optim.Adam(self.agent.parameters(), lr = 0.01)
        self.agent.optimizer = optim_agent
    
    def move(self):
        bird_dead = False
        "Update object positions."
        bird.y -= 5 
        
        for ball in balls: 
            ball.x -= 3 
    
        if rd.randrange(10) == 0: 
            y = rd.randrange(-199, 199) 
            ball = vector(199, y) 
            balls.append(ball) 
        
        while len(balls) > 0 and not inside(balls[0]):  
            balls.pop(0) 
            
        if not inside(bird): 
            bird_dead = True    
            draw(bird_dead) 
            score.append(-1)
    
        for ball in balls:
            if abs(ball - bird) < 15: 
                bird_dead = True    
                draw(bird_dead) 
                score.append(-1)
        
        pos_bird = np.array([bird.x, bird.y]) 
        pos_bird_norm, dis_wall_norm = scale_input(pos_bird) 
        
        #Binary Map for Balls
        bin_map = np.zeros((400, 200))
        for ball in balls:
            x = ball.x
            y = ball.y + 200
            
            if x > 0:
                bin_map[[y],[x]] = 1
        
        bin_map_vector = np.reshape(bin_map, bin_map.shape[0] * bin_map.shape[1])             
        
        #Create Input Data for Network
        data = np.append(np.append(pos_bird_norm, dis_wall_norm), bin_map_vector) 
        data = torch.from_numpy(data).float()
        
        #Feedforward Path
        nn_output = self.agent(data)
        jump = binary(nn_output)
        
        if jump == 1:
            tap() 
               
        score.append(1) 
        draw(bird_dead) 
        sleep(0.05)
        
        '''
        #Calculate Reward
        
        #Train Agent Backpropagation 
        self.agent.optim_agent.zero_grad() 
        loss_P_train.backward() 
        self.agent.optim_agent.step()
        '''
        
        return bird_dead


###########################################
###########################################
for nr_games in range(5):
    
    #Initialization 
    training_framework = Training_Framework()
    
    bird = vector(0, 0) 
    balls = [] 
    score = []
    bird_dead = False

    tl.setup(420, 420) 
    tl.hideturtle()
    tl.up()
    tl.tracer(False)
    
    while bird_dead == False:
        
        bird_dead = training_framework.move()
        
        if bird_dead == True:
            break
        
    print(sum(score))
        
    tl.bye()


    

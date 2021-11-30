import random as rd
import turtle as tl
import numpy as np
import torch
from matplotlib import pyplot as plt
from freegames import vector
from time import sleep
from Agent import Agent

input_size = 6
output_size = 2

agent = Agent(input_size, output_size) 
agent.load_state_dict(torch.load('agent_nn'))
agent.eval()

###########################################
def tap(bird):
    "Move bird up in response to screen tap."
    up = vector(0, 20) 
    bird.move(up)

###########################################
def inside(point):
    "Return True if point on screen."
    return 0 <= point.x < 200 and -200 < point.y < 200 

###########################################
def draw(bird, balls_low, balls_high, bird_dead):
    "Draw screen objects."
    tl.clear()                          
    
    tl.goto(bird.x, bird.y)             

    if bird_dead:
        tl.dot(10, 'red')
    else:
        tl.dot(10, 'green')

    for ball in balls_low:
        tl.goto(ball.x, ball.y)
        tl.dot(20, 'black')
        
    for ball in balls_high:
        tl.goto(ball.x, ball.y)
        tl.dot(20, 'black')

    tl.update() 
    
###########################################
def move(bird, balls_low, balls_high, action, step_count, score):
    bird_dead = False
    step_count += 1
    vertical_bird_distance = 200

    pos_y_bird = bird.y
    pos_y_bird_norm = (pos_y_bird-(-199))/(199-(-199))
    
    pos_y_wall_low = 0
    pos_y_wall_low_norm = pos_y_wall_low
    pos_y_wall_high = 1
    pos_y_wall_high_norm = pos_y_wall_high
    
    pos_x_next_ball_low = balls_low[0].x
    pos_x_next_ball_low_norm = (pos_x_next_ball_low-(0))/(199-(0))
    pos_y_next_ball_low = balls_low[0].y
    pos_y_next_ball_low_norm = (pos_y_next_ball_low-(-199))/(199-(-199))
    pos_y_next_ball_high = balls_high[0].y
    pos_y_next_ball_high_norm = (pos_y_next_ball_high-(-199))/(199-(-199))
    
    state = [pos_y_bird_norm, pos_y_wall_low_norm, pos_y_wall_high_norm, pos_x_next_ball_low_norm, pos_y_next_ball_low_norm, pos_y_next_ball_high_norm]
    state = np.reshape(state, [1, input_size])    

    "Update object positions."
    bird.y -= 5                         
    
    for ball in balls_low:              
        ball.x -= 3                     
        
    for ball in balls_high:             
        ball.x -= 3                     

    if step_count % 30 == 0:           
        y1 = rd.randrange(-150, -50)    
        ball_low = vector(199, y1)     
        balls_low.append(ball_low)       
        y2 = y1 + vertical_bird_distance
        ball_high = vector(199, y2) 
        balls_high.append(ball_high)
    
    while len(balls_low) > 0 and not inside(balls_low[0]): 
        "Solange Anzahl Raben > 0 UND Wenn Raben nicht mehr auf Screen" 
        balls_low.pop(0)                
        balls_high.pop(0)               
        score += 1
        
    if not inside(bird):                
        bird_dead = True    
        draw(bird, balls_low, balls_high, bird_dead)    

    for ball in balls_low:             
        if abs(bird.x - ball.x) < 10:
            if (bird.y - ball.y) < 10:  
                bird_dead = True    
                draw(bird, balls_low, balls_high, bird_dead)
            
    for ball in balls_high:             
        if abs(bird.x - ball.x) < 10:
            if (bird.y - ball.y) > -10: 
                bird_dead = True    
                draw(bird, balls_low, balls_high, bird_dead)
 
    jump = action    
    if jump == 1:
        tap(bird) 
           
    draw(bird, balls_low, balls_high, bird_dead)
    sleep(0.03)
    
    return bird, balls_low, balls_high, state, step_count, bird_dead, score


###########################################
def main():
    nr_games = 100
    score_append = np.array([])
    
    for i in range(nr_games):
        
        bird = vector(0, 0)                    
        balls_low = [vector(199, -100)]        
        balls_high = [vector(199, 100)]       
        bird_dead = False
        step_count = 1
        score = 0
        
        tl.setup(420, 420)                      
        tl.hideturtle()
        tl.up()
        tl.tracer(False)
        
        state = [0.5, 0.5, 1.0, 1.0, 0.24874372, 0.75125628] 
        state = np.reshape(state, [1, input_size])
        
        while bird_dead == False:
            action = agent.act(state) 
              
            bird, balls_low, balls_high, state, step_count, bird_dead, score = move(bird, balls_low, balls_high, action, step_count, score)
            
            if bird_dead == True:            
                tl.bye()
                break
            
            if score == 1000:
                break
     
        print("Epoch: ", i)
        print("Score: ", score)
       
        score_append = np.append(score_append, score)
    
    length = len(score_append)
    x_values = np.array([])
    for i in range(length):
        x_values = np.append(x_values, i + 1)
        
    print("##############")
    print("Game:  ", i+1)
    print("Score: ", score)
    
    #Result Plot Score
    fig1, ax1 = plt.subplots()
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Score')
    ax1.scatter(x_values, score_append, color='blue')
    ax1.tick_params(axis='y', labelcolor = 'blue')
    fig1.tight_layout()
    plt.show()
      

###########################################
if __name__ == "__main__":
    main()

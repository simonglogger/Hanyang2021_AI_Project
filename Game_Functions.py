import random as rd
import turtle as tl
import numpy as np
from freegames import vector
from time import sleep

input_size = 6
output_size = 2

###########################################
def tap(bird):
    "Move bird up in response to action"
    up = vector(0, 20) 
    bird.move(up)

###########################################
def inside(point):
    "Return true if point on screen."
    return 0 <= point.x < 200 and -200 < point.y < 200 

###########################################
def draw(bird, balls_low, balls_high, bird_dead):
    "Draw game map and objects"
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
    vertical_bird_distance = 200
    
    'Reward with regard to birds vertical distance to the center of next balls pair'
    center_balls = balls_high[0].y - vertical_bird_distance/2
    if bird.y >= 0:
        distance = abs(bird.y - center_balls)
    elif bird.y < 0:
        distance = abs(center_balls - bird.y)
    distance_norm = (distance-(0))/(vertical_bird_distance/2-(0))
    
    'distance_norm = 0 --> center, distance_norm = 1 --> position ball'
    'distance_norm > 1 --> outside balls'
    reward = (0.8 - distance_norm)/10
    
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

    "Update object positions"
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
        reward = 1
        score += 1
        
    if not inside(bird):                
        bird_dead = True    
        #draw(bird, balls_low, balls_high, bird_dead)   
        reward = -10

    for ball in balls_low:              
        if abs(bird.x - ball.x) < 10:
            if (bird.y - ball.y) < 10:  
                bird_dead = True    
                #draw(bird, balls_low, balls_high, bird_dead)
                reward = -10
            
    for ball in balls_high:             
        if abs(bird.x - ball.x) < 10:
            if (bird.y - ball.y) > -10: 
                bird_dead = True    
                #draw(bird, balls_low, balls_high, bird_dead)
                reward = -10
 
    jump = action    
    if jump == 1:
        tap(bird) 
           
    #draw(bird, balls_low, balls_high, bird_dead)
    #sleep(0.03)
    
    "Define next state - normalized"
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
    
    next_state = [pos_y_bird_norm, pos_y_wall_low_norm, pos_y_wall_high_norm, pos_x_next_ball_low_norm, pos_y_next_ball_low_norm, pos_y_next_ball_high_norm]
    next_state = np.reshape(next_state, [1, input_size])
    
    return state, action, reward, next_state, bird_dead, score


###########################################
def main(bird, balls_low, balls_high, action, step_count, score):
            
    state, action, reward, next_state, bird_dead, score = move(bird, balls_low, balls_high, action, step_count, score)
            
    #if bird_dead == True:            
        #tl.bye()
        
    return state, action, reward, next_state, bird_dead, score
      

###########################################
if __name__ == "__main__":
    main()
import random as rd
import turtle as tl
import numpy as np
from freegames import vector
from time import sleep

input_size = 5
output_size = 2

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
    tl.clear()                          #Clear Screen
    
    tl.goto(bird.x, bird.y)             #Bewegt Vogel auf absolute Position

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
def move(bird, balls_low, balls_high, action, step_count):
    bird_dead = False
    reward = 0.1
    
    "Define state - normalized"
    pos_y_bird = bird.y
    pos_y_bird_norm = (pos_y_bird-(-199))/(199-(-199))
    
    diff_y_wall = 199 - abs(bird.y)
    diff_y_wall_norm = (diff_y_wall-(0))/(199-(0))
    
    pos_x_next_ball_low = balls_low[0].x
    pos_x_next_ball_low_norm = (pos_x_next_ball_low-(0))/(199-(0))
    pos_y_next_ball_low = balls_low[0].y
    pos_y_next_ball_low_norm = (pos_y_next_ball_low-(-199))/(199-(-199))
    pos_y_next_ball_high = balls_high[0].y
    pos_y_next_ball_high_norm = (pos_y_next_ball_high-(-199))/(199-(-199))
    
    state = [pos_y_bird_norm, diff_y_wall_norm, pos_x_next_ball_low_norm, pos_y_next_ball_low_norm, pos_y_next_ball_high_norm]
    state = np.reshape(state, [1, input_size])
    
    "Update object positions."
    bird.y -= 5                         #Fallgeschwindigkeit Bird
    
    for ball in balls_low:              #Von Rabe 1 bis Anzahl Raben
        ball.x -= 3                     #Alle Raben um x = -3 bewegen
        
    for ball in balls_high:             #Von Rabe 1 bis Anzahl Raben
        ball.x -= 3                     #Alle Raben um x = -3 bewegen

    if step_count % 30 == 0:            #Rabe wird alle 10 Steps initialisiert
        y1 = rd.randrange(-150, -50)     #Ein Rabe irgendwo im Bereich y zwischen +/-199
        ball_low = vector(199, y1)      #Rabe auf Flüghöhe definiert mit Start x bei 199
        balls_low.append(ball_low)      #Neuer Rabe wird Ball Matrix hinzugefügt  
        y2 = y1 + 200
        ball_high = vector(199, y2) 
        balls_high.append(ball_high)
    
    while len(balls_low) > 0 and not inside(balls_low[0]): 
        "Solange Anzahl Raben > 0 UND Wenn Raben nicht mehr auf Screen" 
        balls_low.pop(0)                #Entfernt Rabe bei Index 0 von Liste
        balls_high.pop(0)
        reward = 1
        
    if not inside(bird):                #Wenn Vogel außerhalb Bildschirm
        bird_dead = True    
        draw(bird, balls_low, balls_high, bird_dead)    #Färbt Vogel rot
        reward = -10

    for ball in balls_low:              #Von Rabe 1 bis Anzahl Raben
        if abs(bird.x - ball.x) < 10:
            if (bird.y - ball.y) < 10:  #Wenn Vogel unter low
                bird_dead = True    
                draw(bird, balls_low, balls_high, bird_dead)
                reward = -10
            
    for ball in balls_high:             #Von Rabe 1 bis Anzahl Raben
        if abs(bird.x - ball.x) < 10:
            if (bird.y - ball.y) > -10: #Wenn Vogel über high
                bird_dead = True    
                draw(bird, balls_low, balls_high, bird_dead)
                reward = -10
 
    jump = action    
    if jump == 1:
        tap(bird) 
           
    draw(bird, balls_low, balls_high, bird_dead)
    sleep(0.001)
    
    "Define next state - normalized"
    pos_y_bird = bird.y
    pos_y_bird_norm = (pos_y_bird-(-199))/(199-(-199))
    
    diff_y_wall = 199 - abs(bird.y)
    diff_y_wall_norm = (diff_y_wall-(0))/(199-(0))
    
    pos_x_next_ball_low = balls_low[0].x
    pos_x_next_ball_low_norm = (pos_x_next_ball_low-(0))/(199-(0))
    pos_y_next_ball_low = balls_low[0].y
    pos_y_next_ball_low_norm = (pos_y_next_ball_low-(-199))/(199-(-199))
    pos_y_next_ball_high = balls_high[0].y
    pos_y_next_ball_high_norm = (pos_y_next_ball_high-(-199))/(199-(-199))
    
    next_state = [pos_y_bird_norm, diff_y_wall_norm, pos_x_next_ball_low_norm, pos_y_next_ball_low_norm, pos_y_next_ball_high_norm]
    next_state = np.reshape(next_state, [1, input_size])
    
    return state, action, reward, next_state, bird_dead


###########################################
def main(bird, balls_low, balls_high, action, step_count):
            
    state, action, reward, next_state, bird_dead = move(bird, balls_low, balls_high, action, step_count)
            
    if bird_dead == True:            
        tl.bye()
        
    return state, action, reward, next_state, bird_dead
      

###########################################
if __name__ == "__main__":
    main()
"""Flappy, game inspired by Flappy Bird.

Exercises

1. Keep score.
2. Vary the speed.
3. Vary the size of the balls.
4. Allow the bird to move forward and back.

"""

import random as rd
import turtle as tl

from freegames import vector

bird = vector(0, 0) #Initiale Position Vogel
balls = [] #Leeres array für alle Raben (Positionen)
score = []


#def tap(x, y):
def tap():
    "Move bird up in response to screen tap."
    up = vector(0, 30) #Bird wird 30 Einheiten nach oben bewegt auf Mausklick
    bird.move(up)


def inside(point):
    "Return True if point on screen."
    return -200 < point.x < 200 and -200 < point.y < 200 #Gibt 1 wenn Bird auserhalb Spielfeld


def draw(alive):
    "Draw screen objects."
    tl.clear() #Clear Screen

    tl.goto(bird.x, bird.y) #Bewegt Vogel auf absolute Position

    if alive:
        tl.dot(10, 'green')
    else:
        tl.dot(10, 'red')

    for ball in balls:
        tl.goto(ball.x, ball.y)
        tl.dot(20, 'black')

    tl.update()
    
   
############################### 
def rand():
    if rd.randrange(5) == 0:
        y = 1
    else:
        y = 0
        
    return y  
###############################
    

def move():
    "Update object positions."
    bird.y -= 5 #Fallgeschwindigkeit Bird

    for ball in balls: #Von Rabe 1 bis Anzahl Raben
        ball.x -= 3 #Alle Raben um x = -3 bewegen

    if rd.randrange(10) == 0: #Zufall wann Raben kommen, etwa alle 10 Einheiten
        y = rd.randrange(-199, 199) #Ein Rabe irgendwo im Bereich y zwischen +/-199
        ball = vector(199, y) #Rabe auf Flüghöhe definiert mit Start x bei 199
        balls.append(ball) #Neuer Rabe wird Ball Matrix hinzugefügt

    while len(balls) > 0 and not inside(balls[0]): #Solange Anzahl Raben > 0 UND Rabe nicht mehr auf Screen 
        balls.pop(0) #Entfernt Rabe bei Index 0 von Liste

    if not inside(bird): #Wenn Vogel außerhalb Bildschirm
        draw(False) #Färbt Vogel rot
        ################
        score.append(-1)
        ################
        return

    for ball in balls: #Von Rabe 1 bis Anzahl Raben
        if abs(ball - bird) < 15: #Wenn Abstand < 15 Einheiten
            draw(False) #Färbt Vogel rot
            ################
            score.append(-1)
            ################
            return
        
        
###################################    
    jump = rand()    
    if jump == 1:
       tap() 
       
    score.append(1)
###################################
    
    
    draw(True) #Wenn kein Kontakt: Vogel bleibt grün
    tl.ontimer(move, 50) #Timer der Funktion alle x Millisekunden auslöst.


tl.setup(420, 420, 370, 0) #Spielfeld Definition (Breite, Höhe, Position Spielfeld X, Position Spielfeld Y)
tl.hideturtle()
tl.up()
tl.tracer(False)
#tl.onscreenclick(tap)
move()
tl.done()
print(score)
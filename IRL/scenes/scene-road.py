import sys
import random
import math
import numpy as np
import os

import pygame
from pygame.color import THECOLORS
from pygame.locals import * # for macro "QUIT"

import pymunk
from pymunk.vec2d import Vec2d
import pymunk.pygame_util


# set up pygame 
screenX = 1200
screenY = 100
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (screenX,screenY)
multi = 5 # enlarge the size of simulation scenario
width = 30 * multi
height = 150 * multi
pygame.init()
screen = pygame.display.set_mode((width, height))
screen.set_alpha(None)
clock = pygame.time.Clock()

roadCtr = pymunk.Vec2d(width/2,0)
multiVec = pymunk.Vec2d(multi, multi)


# flags
show_sensors = True
draw_screen = True
    

class GameState:
    def __init__(self):
        # set up simulation
        self.draw_options = pymunk.pygame_util.DrawOptions(screen)
        self.space = pymunk.Space()
        self.space.gravity = pymunk.Vec2d(0, 0) # x and y direction gravity 
        self.simstep = 0.0625 # 16 fps
        self.num_steps = 0
        
        # create roads
        #self.create_road(roadCtr+(0, 0), roadCtr+(0, height))
        #self.create_road(roadCtr+multiVec*(3.75, 0), roadCtr+multiVec*(3.75, 150))
        #self.create_road(roadCtr+multiVec*(-3.75, 0), roadCtr+multiVec*(-3.75, 150))
        #self.create_road(roadCtr+multiVec*(6.75, 0), roadCtr+multiVec*(6.75, 150))
        #self.create_road(roadCtr+multiVec*(-6.75, 0), roadCtr+multiVec*(-6.75, 150))
        
        
        # create pedestrains
        #self.create_ped(roadCtr + multiVec*(0, 75))
        
        # create traffic
        self.traffic = []
        self.traffic.append(self.create_car(roadCtr + multiVec*(1.875,30)))
        
        #pygame.draw.lines(screen, THECOLORS["white"], False, [(100,100), (150,200), (200,100)], 1)
        
    
    def frame_step(self, action):		
        # quit simulation
        for event in pygame.event.get():
            if event.type == QUIT:
                sys.exit(0)
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                sys.exit(0)
            
        # draw and update
        self.space.step(self.simstep) 
        screen.fill(THECOLORS["black"])
        self.space.debug_draw(self.draw_options)
        
        # pygame draw
        pygame.draw.line(screen, THECOLORS["white"], roadCtr+(0, 0), roadCtr+(0, height))
        pygame.draw.line(screen, THECOLORS["white"], roadCtr+multiVec*(3.75, 0), roadCtr+multiVec*(3.75, 150))
        pygame.draw.line(screen, THECOLORS["white"], roadCtr+multiVec*(-3.75, 0), roadCtr+multiVec*(-3.75, 150))
        pygame.draw.line(screen, THECOLORS["white"], roadCtr+multiVec*(6.75, 0), roadCtr+multiVec*(6.75, 150))
        pygame.draw.line(screen, THECOLORS["white"], roadCtr+multiVec*(-6.75, 0), roadCtr+multiVec*(-6.75, 150))
        #pygame.draw.circle(screen, (255, 255, 255), (50,200), 2)
        
        pygame.display.flip()
        clock.tick(30)
        
        # advance the simulation step
        self.num_steps += 1
        
                
    ##################################
    ###  create stuff
    ##################################
    # mass: affecs the difficulty of moving
    # moment of inertia (angular mass): affects the difficulty of rotating	
    def create_road(self, startCoords, endCoords):
        road_body = pymunk.Body(pymunk.inf, pymunk.inf, pymunk.Body.STATIC)
        road_shape = pymunk.Segment(road_body, startCoords, endCoords, 1)
        road_shape.color = THECOLORS["white"]
        self.space.add(road_body, road_shape)
        
    def create_ped(self, pos):
        pedMass = 1
        pedRadius = 0.25 * multi
        pedMoment = pymunk.moment_for_circle(pedMass, 0, pedRadius, (0, 0))
        self.pedBody = pymunk.Body(pedMass, pedRadius)
        self.pedBody.position = pos
        self.pedShape = pymunk.Circle(self.pedBody, pedRadius) 
        self.pedShape.color = THECOLORS["white"]
        self.pedShape.elasticity = 1.0
        self.pedShape.angle = 0
        direction = Vec2d(1, 0).rotated(self.pedBody.angle)
        self.space.add(self.pedBody, self.pedShape)	
        
    def create_av(self, pos, theta):
        avMass = 1
        avRadius = 25
        self.avBody = pymunk.Body(avMass, pymunk.moment_for_circle(avMass, 0, avRadius, (0, 0)))
        self.avBody.position = pos
        self.avShape = pymunk.Circle(self.avBody, avRadius)
        self.avShape.color = THECOLORS["green"]
        self.avShape.elasticity = 1.0
        self.avBody.angle = theta
        drivingDir = Vec2d(1, 0).rotated(self.avBody.angle)
        self.avBody.apply_impulse_at_local_point(drivingDir)
        self.space.add(self.avBody, self.avShape)
                
    def create_car(self, pos):
        carMass = 1
        carSize = multiVec * (2.5, 4.5)
        carBody = pymunk.Body(carMass, pymunk.moment_for_box(carMass, carSize))
        carBody.position = pos
        carShape = pymunk.Poly.create_box(carBody, carSize)
        carShape.color = THECOLORS["white"]
        self.space.add(carBody, carShape)
           
           
      
    ##################################
    ###  move stuff
    ##################################
    def move_ped(self):
        speed = random.randint(1, 10)
        #self.cat_body.angle -= random.randint(-1, 1)
        self.cat_body.angle = 2.5
        direction = Vec2d(1, 0).rotated(self.cat_body.angle)
        self.cat_body.velocity = speed * direction
        

    
        
if __name__ == '__main__':
    game_state = GameState()
    while True:
        game_state.frame_step((random.randint(0, 2)))

import sys
import random
import math
import numpy as np

import pygame
from pygame.color import THECOLORS
from pygame.locals import * # for macro "QUIT"

import pymunk
from pymunk.vec2d import Vec2d
import pymunk.pygame_util


# set up pygame 
width = 1000
height = 700
pygame.init()
screen = pygame.display.set_mode((width, height))
screen.set_alpha(None)
clock = pygame.time.Clock()

# flags
show_sensors = True
draw_screen = True
	
class GameState:
	def __init__(self):
		# car status
		self.crashed = False
		self.carspeed = 10
		
		# set up pymunk
		self.draw_options = pymunk.pygame_util.DrawOptions(screen)
		self.space = pymunk.Space()
		self.space.gravity = pymunk.Vec2d(0, 0) # x and y direction gravity 
		self.simstep = 0.0625 # 16 FPS 
		
		# record steps
		self.num_steps = 0
		
		# create walls
		self.create_wall((0, 1), (0, height))
		self.create_wall((1, height), (width, height))
		self.create_wall((width-1, height), (width-1, 1))
		self.create_wall((1, 1), (width, 1))
		
		# create obstacles
		self.obstacles = []
		self.obstacles.append(self.create_obstacle(200,350,100))
		self.obstacles.append(self.create_obstacle(700,200,125))
		self.obstacles.append(self.create_obstacle(600,600,35))
		
		# create a cat
		self.create_cat()
		
		# create a car
		self.create_car(100, 100, -1)
		self.create_obst_car(300, 100)
		
				
	def frame_step(self, action):
		# move the car according to "action" which could be 0, 1, or 2
		if action == 0: # Turn left
			self.car_body.angle -= .2
		elif action == 1: # Turn right
			self.car_body.angle += .2
		driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
		self.car_body.velocity = self.carspeed * driving_direction
			
		# move the obstacles
		if self.num_steps % 100 == 0:
			self.move_obstacles()
			
		# move the cat
		if self.num_steps % 5 == 0:
			self.move_cat()
			
		# quit the game
		for event in pygame.event.get():
			if event.type == QUIT:
				sys.exit(0)
			elif event.type == KEYDOWN and event.key == K_ESCAPE:
				sys.exit(0)
			
		# draw and update
		self.space.step(self.simstep) 
		screen.fill(THECOLORS["black"])
		self.space.debug_draw(self.draw_options)
		pygame.display.flip()
		#print("tick " + str(pygame.time.get_ticks()))
		#print(clock.get_fps())
		clock.tick(30)
		
		# get the current location and reading
		x, y = self.car_body.position
		readings = self.get_sonar_readings(x, y, self.car_body.angle)
		
		# normalize the reading, because for each sonar we have 40 detection points
		normalized_readings = [(x-20.0)/20.0 for x in readings] 
		
		# treat the normalized readings as the state of the car 
		state = np.array([normalized_readings])

		# set the reward.
		# car crashed when any reading == 1 (meaning the first detection point of an arm has detected an obstacle)
		if self.car_is_crashed(readings):
			self.crashed = True
			reward = -500
			self.recover_from_crash(driving_direction)
		else:
			# higher readings are better, so return the sum.
			reward = -5 + int(self.sum_readings(readings) / 10)
		
		self.num_steps += 1
		
		return reward, state
		
	
	# mass: affecs the difficulty of moving
	# moment of inertia (angular mass): affects the difficulty of rotating	
	def create_wall(self, a, b):
		#wall_mass = 1
		#wall_moment = pymunk.moment_for_segment(wall_mass, a, b, 1)
		wall_body = pymunk.Body(pymunk.inf, pymunk.inf, pymunk.Body.STATIC)
		wall_shape = pymunk.Segment(wall_body, a, b, 1)
		wall_shape.elasticity = 1.0
		wall_shape.color = THECOLORS["red"]
		self.space.add(wall_body, wall_shape)
		
		
	def create_obstacle(self, x, y, r):
		obst_body = pymunk.Body(pymunk.inf, pymunk.inf, pymunk.Body.KINEMATIC)
		obst_shape = pymunk.Circle(obst_body, r)
		obst_shape.elasticity = 1.0
		obst_body.position = x, y
		obst_shape.color = THECOLORS["blue"]
		self.space.add(obst_body, obst_shape)
		return obst_body
		
		
	def create_cat(self):
		cat_mass = 1
		cat_radius = 30
		cat_inertia = pymunk.moment_for_circle(cat_mass, 0, cat_radius, (0, 0))
		
		self.cat_body = pymunk.Body(cat_mass, cat_inertia)
		self.cat_body.position = 50, height - 100
		self.cat_shape = pymunk.Circle(self.cat_body, cat_radius) 
		self.cat_shape.color = THECOLORS["orange"]
		self.cat_shape.elasticity = 1.0
		self.cat_shape.angle = 3
		direction = Vec2d(1, 0).rotated(self.cat_body.angle)
		self.space.add(self.cat_body, self.cat_shape)
		
		
	def create_car(self, x, y, theta):
		car_mass = 1
		car_inertia = pymunk.moment_for_circle(car_mass, 0, 14, (0, 0))
		self.car_body = pymunk.Body(car_mass, car_inertia)
		self.car_body.position = x, y
		self.car_shape = pymunk.Circle(self.car_body, 25)
		self.car_shape.color = THECOLORS["green"]
		self.car_shape.elasticity = 1.0
		self.car_body.angle = theta
		driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
		self.car_body.apply_impulse_at_local_point(driving_direction)
		self.space.add(self.car_body, self.car_shape)
		
		
	def create_obst_car(self, x, y):
		car_mass = 1
		car_size = (30, 50)
		car_moment = pymunk.moment_for_box(car_mass, car_size)
		self.obst_car_body = pymunk.Body(car_mass, car_moment)
		self.obst_car_body.position = x, y
		self.obst_car_shape = pymunk.Poly.create_box(self.obst_car_body, car_size)
		self.obst_car_shape.color = THECOLORS["white"]
		self.space.add(self.obst_car_body, self.obst_car_shape)
		
		
	def move_obstacles(self):
		for obstacle in self.obstacles:
			#speed = random.randint(1, 5)
			speed = 1
			direction = Vec2d(1, 0).rotated(self.car_body.angle + random.randint(-2, 2))
			obstacle.velocity = speed * direction
           
            
	def move_cat(self):
		speed = random.randint(1, 10)
		self.cat_body.angle -= random.randint(-1, 1)
		direction = Vec2d(1, 0).rotated(self.cat_body.angle)
		self.cat_body.velocity = speed * direction
		
		
	def car_is_crashed(self, readings):
		if readings[0] == 1 or readings[1] == 1 or readings[2] == 1:
			return True
		else:
			return False
            
      
	def recover_from_crash(self, driving_direction):
		while self.crashed:
			# go backwards 
			self.car_body.velocity = -self.carspeed * driving_direction
			self.crashed = False
			for i in range(10):
				self.car_body.angle += .2  # Turn a little.
				screen.fill(THECOLORS["red"])  # Shine the screen
				self.space.debug_draw(self.draw_options)
				self.space.step(self.simstep)
				if draw_screen:
					pygame.display.flip()
				clock.tick()
    		
    	
	def sum_readings(self, readings):
		'''
		Sum the number of non-zero readings
		'''
		tot = 0
		for i in readings:
			tot += i
		return tot	
        
        
	def get_sonar_readings(self, x, y, angle):
		'''
		Sonar readings return the "distance" in each sonar arm.
		For example: 
		arm_left reading: 0, 0, 0, 0, 0, 1, 1, 1, 1
		then arm_left returns a distance of 6 (the first non-zero reading of this arm)
		'''
		# make our arms
		arm_left = self.make_sonar_arm(x, y)
		arm_middle = arm_left
		arm_right = arm_left

		# rotate them and get readings
		readings = []
		readings.append(self.get_arm_distance(arm_left, x, y, angle, 0.75))
		readings.append(self.get_arm_distance(arm_middle, x, y, angle, 0))
		readings.append(self.get_arm_distance(arm_right, x, y, angle, -0.75))
		
		if show_sensors:
			pygame.display.update()

		return readings
        
        
	def get_arm_distance(self, arm, x, y, angle, offset):
		# look at each detection point and see if we've hit something
		detect_idx = 0
		for point in arm:
			detect_idx += 1

			# move the point to the right spot.
			rotated_p = self.get_rotated_point(x, y, point[0], point[1], angle + offset)

			# check the rotated detection point if hit anything
			# return the current detect_idx if yes
			if rotated_p[0] <= 0 or rotated_p[1] <= 0 or rotated_p[0] >= width or rotated_p[1] >= height:
				return detect_idx  # the detection point is off the screen.
			else:
				obs = screen.get_at(rotated_p) # get the color of a point 
				if self.get_track_or_not(obs) != 0:
					return detect_idx

			if show_sensors:
				pygame.draw.circle(screen, (255, 255, 255), (rotated_p), 2)
				
		return detect_idx
           
           
	def get_rotated_point(self, x_1, y_1, x_2, y_2, angle):
		# Rotate x_2, y_2 around x_1, y_1 by the angle in radians
		x_change = (x_2 - x_1) * math.cos(angle) + (y_2 - y_1) * math.sin(angle)
		y_change = (y_1 - y_2) * math.cos(angle) - (x_1 - x_2) * math.sin(angle)
		new_x = x_change + x_1
		new_y = height - (y_change + y_1)
		return int(new_x), int(new_y)

    
	def get_track_or_not(self, reading):
		if reading == THECOLORS['black']:
			return 0
		else:
			return 1
            
        
	def make_sonar_arm(self, x, y):
		'''
		Build a flat sonar arm
		'''
		spread = 10  # Default spread.
		init_placement = 20  # Gap before first sensor.
		arm = []
		for i in range(1, 40):
			arm.append((init_placement + x + (spread * i), y)) # first sensor - [init_placement + x + 10, y], second sensor - [init_placement + x + 20, y], etc.
		return arm
	
		
if __name__ == '__main__':
	game_state = GameState()
	while True:
		game_state.frame_step((random.randint(0, 2)))

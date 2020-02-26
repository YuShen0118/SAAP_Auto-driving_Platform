import sys
import random
import math
import numpy as np
from ctypes import *
import ctypes

import pygame
from pygame.color import THECOLORS
from pygame.locals import * # for macro "QUIT"

import pymunk
from pymunk.vec2d import Vec2d
import pymunk.pygame_util

# PyGame init
width = 1000
height = 700
pygame.init()
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

# Turn off alpha since we don't use it.
screen.set_alpha(None)

# Showing sensors and redrawing slows things down.
flag = True
show_sensors = flag
draw_screen = flag

MULTI = 5 # enlarge the size of simulation scenario
multiVec = pymunk.Vec2d(MULTI, MULTI)
offset = 70


class Vector2(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float)]


class GameState:
    def __init__(self, weights='', scene_file_name='', state_num=46, use_expert=True):
        if use_expert:
            # for windows
            self.expert_lib = ctypes.WinDLL ("./3thPartLib/RVO2/RVO_warper.dll")
        self.driving_history = []
        self.max_history_num = 50

        # Global-ish.
        self.crashed = False

        # set weights for calculating the reward function
        if len(weights) == state_num:
            self.W = weights 
        else:
            #self.W = [1, 1, 1, 1, 1, 1, 1, 1] # randomly assign weights if not provided
            self.W = np.ones(state_num)

        # Physics stuff.
        self.draw_options = pymunk.pygame_util.DrawOptions(screen)
        self.space = pymunk.Space()
        self.space.gravity = pymunk.Vec2d(0., 0.)

        c_handler = self.space.add_wildcard_collision_handler(2)
        c_handler.begin = self.collision_callback
        
        #TODO, fix number of other vehicles
        c_handler3 = self.space.add_wildcard_collision_handler(3)
        c_handler3.begin = self.collision_callback3
        c_handler33 = self.space.add_collision_handler(3, 3)
        c_handler33.begin = self.collision_callback33
        
        self.simstep = 0.02 # 50 FPS 
        #self.simstep = 0.0625 # 16 FPS 
        
        self.steer_section_number = 5
        self.steer_zero_section_no = 2
        self.steer_per_section = math.pi / 4

        self.acc_section_number = 5
        self.acc_zero_section_no = 1
        self.acc_per_section = 5

        self.preferred_speed = 10.0       # in meter, TODO
        self.max_speed = 20.0 * MULTI

        # Create the car.
        self.create_main_car((80+offset)*MULTI, (0+offset)*MULTI, 15)

        # Record steps.
        self.num_steps = 0
        self.num_obstacles_type = 4

        # Create walls.
        static = [
            pymunk.Segment(
                self.space.static_body,
                (0, 1), (0, height), 1),
            pymunk.Segment(
                self.space.static_body,
                (1, height), (width, height), 1),
            pymunk.Segment(
                self.space.static_body,
                (width-1, height), (width-1, 1), 1),
            pymunk.Segment(
                self.space.static_body,
                (1, 1), (width, 1), 1)
        ]
        for s in static:
            s.friction = 1.
            s.group = 1
            s.collision_type = 1
            s.color = THECOLORS['blue']
        self.space.add(static)

        # Create some obstacles, semi-randomly.
        # We'll create three and they'll move around to prevent over-fitting.
        self.obstacles = []
        self.obstacles_poly = []
        #self.obstacles.append(self.create_obstacle(380, 220, 70, "yellow"))
        #self.obstacles.append(self.create_obstacle(250, 500, 70, "yellow"))
        #self.obstacles.append(self.create_obstacle(780, 330, 70, "brown"))
        #self.obstacles.append(self.create_obstacle(530, 500, 70, "brown"))

        self.obstacles_car = []

        obstacle_color = "brown"
        minx = 99999999999
        maxx = 0
        miny = 99999999999
        maxy = 0
        
        if scene_file_name == 'scenes/scene-ground-car.txt':

            self.goals = [[80,0], [50,44],[-45,46],[-52,-45],[-10,-66],[-3,40]] #,[80,-60]
            for goal in self.goals:
                goal[0] = (goal[0] + offset) * MULTI 
                goal[1] = (goal[1] + offset) * MULTI 
            self.current_goal_id = 0
            
            self.pre_goal_dist = (self.goals[0] - self.car_body.position).length

            grid = 5
            border = 20
            for j in range(grid):
                for k in range(grid):
                    lowx = 0 + width/grid*j + border
                    highx = lowx + width/grid - border
                    lowy = 0 + height/grid*k + border
                    highy = lowy + height/grid - border

                    #lowx = minx + (maxx-minx)/grid*j + border
                    #highx = lowx + (maxx-minx)/grid - border
                    #lowy = miny + (maxy-miny)/grid*j + border
                    #highy = lowy + (maxy-miny)/grid - border
                    self.obstacles_car.append(self.create_obstacle_car(
                        np.random.randint(lowx, highx), np.random.randint(lowy, highy), 
                        np.random.random()*math.pi, np.random.random()*20, 3))
        elif scene_file_name!='':
            #load scene
            with open(scene_file_name) as f:
                obj_num = int(f.readline())
                for i in range(obj_num):
                    poly = []
                    pt_num = int(f.readline())
                    for j in range(pt_num):
                        line = f.readline()
                        pt = []
                        for s in line.split(' '):
                            pt.append(MULTI*(float(s)+offset))
                            #pt.append((float(s)+0))
                        #poly.append((pt[0],pt[1]))
                        poly.append(pt)
                        
                        #pygame.draw.circle(screen, (255, 255, 255), (int(pt[0]), height-int(pt[1])), 4)
                        #pygame.display.update()
                        #pygame.event.get()
                        
                        minx = min(minx, pt[0])
                        maxx = max(maxx, pt[0])
                        miny = min(miny, pt[1])
                        maxy = max(maxy, pt[1])

                    self.obstacles.append(self.create_obstacle_poly(poly, obstacle_color))
                    self.obstacles_poly.append(poly)
                    
                    #screen.fill(THECOLORS["black"])
                    #self.space.debug_draw(self.draw_options)
                    
                f.readline()
                obs_car_num = int(f.readline())
                for i in range(obs_car_num):
                    line = f.readline()
                    info = []
                    for s in line.split(' '):
                        info.append(s)
                    self.obstacles_car.append(self.create_obstacle_car(MULTI*(float(info[0])+offset), 
                                                    MULTI*(float(info[1])+offset), 
                                                    float(info[2])/180*math.pi, float(info[3]), 3))

            segment_radius = 5
            poly_tmp = self.create_poly_from_vertical_segment([minx, miny], [minx, maxy], segment_radius)
            self.obstacles.append(self.create_obstacle_poly(poly_tmp, obstacle_color))
            
            poly_tmp = self.create_poly_from_vertical_segment([maxx, miny], [maxx, maxy], segment_radius)
            self.obstacles.append(self.create_obstacle_poly(poly_tmp, obstacle_color))
            
            poly_tmp = self.create_poly_from_horizontal_segment([minx, miny], [maxx, miny], segment_radius)
            self.obstacles.append(self.create_obstacle_poly(poly_tmp, obstacle_color))
            
            poly_tmp = self.create_poly_from_horizontal_segment([minx, maxy], [maxx, maxy], segment_radius)
            self.obstacles.append(self.create_obstacle_poly(poly_tmp, obstacle_color))
            
            
            #self.obstacles.append(self.create_obstacle([minx, miny], [minx, maxy], segment_radius, "brown"))
            #self.obstacles.append(self.create_obstacle([minx, maxy], [maxx, maxy], segment_radius, "brown"))
            #self.obstacles.append(self.create_obstacle([maxx, maxy], [maxx, miny], segment_radius, "brown"))
            #self.obstacles.append(self.create_obstacle([maxx, miny], [minx, miny], segment_radius, "brown"))

            self.goals = [[80,0], [50,44],[-45,46],[-52,-45],[-10,-66],[-3,40]] #,[80,-60]
            for goal in self.goals:
                goal[0] = (goal[0] + offset) * MULTI 
                goal[1] = (goal[1] + offset) * MULTI 
            self.current_goal_id = 0
            
            self.pre_goal_dist = (self.goals[0] - self.car_body.position).length
        else:
            #default scene
            self.obstacles.append(self.create_obstacle([100, 100], [100, 585] , 7, "yellow"))
            self.obstacles.append(self.create_obstacle([450, 600], [100, 600] , 7, "yellow"))
            self.obstacles.append(self.create_obstacle([900, 100], [900, 585] , 7, "yellow"))
            self.obstacles.append(self.create_obstacle([900, 600], [550, 600] , 7, "yellow"))

            self.obstacles.append(self.create_obstacle([200, 100], [200, 480] , 7, "yellow"))
            self.obstacles.append(self.create_obstacle([450, 500], [200, 500] , 7, "yellow"))
            self.obstacles.append(self.create_obstacle([800, 100], [800, 480] , 7, "yellow"))
            self.obstacles.append(self.create_obstacle([800, 500], [550, 500] , 7, "yellow"))

            self.obstacles.append(self.create_obstacle([300, 100], [300, 350] , 7, "yellow"))
            self.obstacles.append(self.create_obstacle([700, 380], [300, 380] , 7, "yellow"))
            self.obstacles.append(self.create_obstacle([700, 100], [700, 350] , 7, "yellow"))

            self.obstacles.append(self.create_obstacle([400, 100], [600, 100] , 7, "brown"))
            self.obstacles.append(self.create_obstacle([400, 200], [600, 200] , 7, "brown"))
            self.obstacles.append(self.create_obstacle([400, 300], [600, 300] , 7, "brown"))

            self.obstacles.append(self.create_obstacle([700, 370], [300, 370] , 7, "brown"))
            self.obstacles.append(self.create_obstacle([310, 100], [310, 350] , 7, "brown"))
            self.obstacles.append(self.create_obstacle([690, 100], [690, 350] , 7, "brown"))

        # Create a cat.
        #self.create_cat()
        
    def collision_callback(self, arbiter, space, data):
        if arbiter.is_first_contact:
            #print('collision!!!')
            arbiter.shapes[0].body.velocity = -arbiter.shapes[0].body.velocity
            arbiter.shapes[1].body.velocity = -arbiter.shapes[1].body.velocity
            self.crashed = True
            return False

    def collision_callback3(self, arbiter, space, data):
        if arbiter.is_first_contact:
            arbiter.shapes[0].body.velocity = -arbiter.shapes[0].body.velocity
            arbiter.shapes[1].body.velocity = -arbiter.shapes[1].body.velocity
            return False

    def collision_callback33(self, arbiter, space, data):
        if arbiter.is_first_contact:
            arbiter.shapes[0].body.velocity = -arbiter.shapes[0].body.velocity
            arbiter.shapes[1].body.velocity = -arbiter.shapes[1].body.velocity
            return False
        
    def create_obstacle(self, xy1, xy2, r, color):
        c_body = pymunk.Body(pymunk.inf, pymunk.inf, pymunk.Body.KINEMATIC)
        #c_shape = pymunk.Circle(c_body, r)
        c_shape = pymunk.Segment(c_body, xy1, xy2, r)
        #c_shape.elasticity = 1.0
        #c_body.position = x, y
        c_shape.friction = 1.
        c_shape.group = 1
        c_shape.collision_type = 1
        c_shape.color = THECOLORS[color]
        self.space.add(c_body, c_shape)
        return c_body
    
    def create_obstacle_poly(self, poly, color):
        c_body = pymunk.Body(pymunk.inf, pymunk.inf, pymunk.Body.KINEMATIC)
        c_shape = pymunk.Poly(c_body, poly)
        c_shape.friction = 1.
        c_shape.group = 1
        c_shape.collision_type = 1
        c_shape.color = THECOLORS[color]
        self.space.add(c_body, c_shape)
        return c_body

    def create_poly_from_vertical_segment(self, xy1, xy2, r):
        # counterclockwise
        poly = []
        poly.append([xy1[0]-r, xy1[1]-r])
        poly.append([xy1[0]+r, xy1[1]-r])
        poly.append([xy2[0]+r, xy2[1]+r])
        poly.append([xy2[0]-r, xy2[1]+r])
        return poly
        
    def create_poly_from_horizontal_segment(self, xy1, xy2, r):
        # counterclockwise
        poly = []
        poly.append([xy1[0]-r, xy1[1]-r])
        poly.append([xy2[0]+r, xy2[1]-r])
        poly.append([xy2[0]+r, xy2[1]+r])
        poly.append([xy1[0]-r, xy1[1]+r])
        return poly

    def create_cat(self):
        inertia = pymunk.moment_for_circle(1, 0, 14, (0, 0))
        self.cat_body = pymunk.Body(1, inertia)
        self.cat_body.position = 50, height - 100
        self.cat_shape = pymunk.Circle(self.cat_body, 30)
        self.cat_shape.color = THECOLORS["orange"]
        self.cat_shape.elasticity = 1.0
        self.cat_shape.angle = 0.5
        direction = Vec2d(1, 0).rotated(self.cat_body.angle)
        self.space.add(self.cat_body, self.cat_shape)

    def create_main_car(self, x, y, r):
        
        carMass = 1
        carSize = multiVec * (4.5, 2.5)
        self.car_body = pymunk.Body(carMass, pymunk.moment_for_box(carMass, carSize))
        self.car_shape = pymunk.Poly.create_box(self.car_body, carSize)
        self.car_shape.collision_type = 2
        
        #inertia = pymunk.moment_for_circle(1, 0, 14, (0, 0))
        #self.car_body = pymunk.Body(1, inertia)
        #self.car_shape = pymunk.Circle(self.car_body, r)

        self.car_body.position = x, y
        self.car_body.init_position = self.car_body.position
        self.car_shape.color = THECOLORS["green"]
        self.car_shape.elasticity = 1.0
        self.car_body.angle = math.pi / 2
        self.car_body.init_angle = self.car_body.angle
        driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
        self.car_body.apply_impulse_at_local_point(driving_direction)
        self.space.add(self.car_body, self.car_shape)
        self.car_reverse_driving = False

        self.car_body.angular_velocity = 0
        self.car_body.velocity = (0,0)
        
    def create_obstacle_car(self, x, y, angle, v, collision_type):
        
        carMass = 1
        carSize = multiVec * (4.5, 2.5)
        car_body = pymunk.Body(carMass, pymunk.moment_for_box(carMass, carSize))
        car_shape = pymunk.Poly.create_box(car_body, carSize)
        car_shape.collision_type = collision_type

        car_body.position = x, y
        car_shape.color = THECOLORS["yellow"]
        car_shape.elasticity = 1.0
        car_body.angle = angle
        driving_direction = Vec2d(1, 0).rotated(car_body.angle)
        car_body.apply_impulse_at_local_point(driving_direction)
        car_body.angular_velocity = 0
        car_body.velocity = v * driving_direction
        
        self.space.add(car_body, car_shape)

        car_reverse_driving = False

        return car_body
        
    def getVecter2List(self, poly):
        vertices=(Vector2*len(poly))()
        i=0
        for vertex in vertices:
            vertex.x = poly[i][0]
            vertex.y = poly[i][1]
            i = i + 1

        return vertices

    def setup_scenario(self, RVO_handler):
        self.expert_lib.RVO_setTimeStep(RVO_handler, c_float(0.02))
    
        velo = self.car_body.velocity
        velocity = Vector2(velo[0], velo[1])
        # Specify the default parameters for agents that are subsequently added.
        self.expert_lib.RVO_setAgentDefaults(RVO_handler, c_float(15.0**MULTI), 10, c_float(1.0), c_float(0.1), c_float(3*MULTI), c_float(20.0*MULTI), velocity)
        pos = self.car_body.position
        position = Vector2(pos[0], pos[1])
        self.expert_lib.RVO_addAgent(RVO_handler, position)
        
        for obstacle in self.obstacles_poly:
            vertexNum = len(obstacle)
            vertices = self.getVecter2List(obstacle)
            self.expert_lib.RVO_addObstacle(RVO_handler, vertexNum, byref(vertices[0]))
            
        id = 0
        for obstacle_car in self.obstacles_car:
            id+=1
            for shape in obstacle_car.shapes:
                poly = []
                for v in shape.get_vertices():
                    poly.append( v.rotated(shape.body.angle) + shape.body.position)
                vertices = self.getVecter2List(poly)
                vertexNum = len(shape.get_vertices())

                #for vertex in vertices:
                #    print(id, vertex.x, vertex.y)
                
                self.expert_lib.RVO_addObstacle(RVO_handler, vertexNum, byref(vertices[0]))


        self.expert_lib.RVO_processObstacles(RVO_handler)
            
    def set_preferred_velocities(self, RVO_handler):
        # Set the preferred velocity to be a vector of unit magnitude (speed) in the
        # direction of the goal.

        current_goal = self.goals[self.current_goal_id]
        agent_id = 0
        position = Vector2(0, 0)
        self.expert_lib.RVO_getAgentPosition(RVO_handler, agent_id, byref(position))
        goalVector = Vector2(current_goal[0] - position.x, current_goal[1] - position.y)

        norm = math.sqrt(goalVector.x*goalVector.x + goalVector.y*goalVector.y)
        if (norm > self.preferred_speed * MULTI):
            goalVector.x = goalVector.x / norm * self.preferred_speed * MULTI
            goalVector.y = goalVector.y / norm * self.preferred_speed * MULTI

        self.expert_lib.RVO_setAgentPrefVelocity(RVO_handler, agent_id, goalVector)
        
    def get_instruction_from_RVO(self, RVO_handler):
        agent_id = 0
        position_new = Vector2(0, 0)
        self.expert_lib.RVO_getAgentPosition(RVO_handler, agent_id, byref(position_new))

        car_direction = self.car_body.angle
        dx = position_new.x - self.car_body.position.x
        dy = position_new.y - self.car_body.position.y
        #target_direction = -math.atan2(dy, dx) + math.pi / 2.0
        target_direction = math.atan2(dy, dx)
        delta_direction = target_direction - car_direction
        while (delta_direction > math.pi):
           delta_direction -= math.pi * 2
        while (delta_direction < -math.pi):
           delta_direction += math.pi * 2
           
        steer_angle = delta_direction / self.simstep
        
        #desired_velocity = Vector2(dx / self.simstep, dy / self.simstep)
        desired_velocity = Vector2(0, 0)
        self.expert_lib.RVO_getAgentVelocity(RVO_handler, agent_id, byref(desired_velocity))
        desired_speed = math.sqrt(desired_velocity.x*desired_velocity.x + desired_velocity.y*desired_velocity.y)
        current_speed = self.car_body.velocity.length
        acceleration = (desired_speed - current_speed) / self.simstep

        return [steer_angle, acceleration]
    
    def get_action_from_instruction(self, instruction):
        [steer_angle, acceleration] = instruction
        steer_action = round(steer_angle / self.steer_per_section) + self.steer_zero_section_no
        acc_action = round(acceleration / self.acc_per_section / MULTI) + self.acc_zero_section_no
        
        steer_action = np.clip(steer_action, 0, self.steer_section_number - 1)
        acc_action = np.clip(acc_action, 0, self.acc_section_number - 1)

        return steer_action * self.acc_section_number + acc_action
    
    def reset_car(self, carPos, carVelo, carAngle):
        self.car_body.position = ((carPos[0] + offset) * MULTI), ((carPos[1] + offset) * MULTI)
        
        self.car_body.velocity = (carVelo[0] * MULTI), (carVelo[1] * MULTI)

        self.car_body.angle = carAngle

    def get_expert_action_out(self, carPos, carVelo, carAngle):
        self.reset_car(carPos, carVelo, carAngle)
        
        RVO_handler = c_longlong(0)
        self.expert_lib.RVO_createEnvironment(byref(RVO_handler))

        self.setup_scenario(RVO_handler);
        self.set_preferred_velocities(RVO_handler)

        self.expert_lib.RVO_doStep(RVO_handler)
        instruction = self.get_instruction_from_RVO(RVO_handler)
        action = self.get_action_from_instruction(instruction)
        
        self.expert_lib.RVO_deleteEnvironment(byref(RVO_handler))
        return action

    def get_expert_action(self):
        RVO_handler = c_longlong(0)
        self.expert_lib.RVO_createEnvironment(byref(RVO_handler))

        self.setup_scenario(RVO_handler);
        self.set_preferred_velocities(RVO_handler)

        self.expert_lib.RVO_doStep(RVO_handler)
        instruction = self.get_instruction_from_RVO(RVO_handler)
        action = self.get_action_from_instruction(instruction)
        
        self.expert_lib.RVO_deleteEnvironment(byref(RVO_handler))
        return action
    
    def get_instruction_from_action_out(self, action):
        [steer_angle, acceleration] = self.get_instruction_from_action(action)
        return [steer_angle, acceleration / MULTI]

    def get_instruction_from_action(self, action):
        #return [action[0] * self.max_steer, action[1] * self.max_acceleration]
        steer_action = int(action / self.acc_section_number)
        acc_action = action % self.acc_section_number

        steer_angle = (steer_action - self.steer_zero_section_no) * self.steer_per_section
        acceleration = (acc_action - self.acc_zero_section_no) * self.acc_per_section * MULTI

        return [steer_angle, acceleration]
    
    def get_reward(self, W, readings, normalize_reading=False, reward_type=1):
        if reward_type == 0:
            reward = 0.4*readings[-3] # get closer to the goal
        
            if (readings[-2] == 1):
                # reach the goal
                reward += 0.5
            if (readings[-1] == 1):
                # collision
                reward -= 0.9

            reward = np.clip(reward, -1, 1)
            if normalize_reading == False:
                reward *= 100
        elif reward_type == 1:
            reward = np.dot(W, readings)
        elif reward_type == 2:
            reward = 0.4*readings[-3] # get closer to the goal
            if (readings[-2] == 1):
                # reach the goal
                reward += 0.5
            if (readings[-1] == 1):
                # collision
                reward -= 0.9
            if normalize_reading == False:
                reward *= 100
                
            reward += np.dot(W, readings)
            
            if normalize_reading:
                reward = np.clip(reward, -1, 1)
        

        return reward
    
    def frame_step(self, action, effect=True, hitting_reaction_mode = 0, normalize_reading = True):
        self.crashed = False
        [steer_angle, acceleration] = self.get_instruction_from_action(action)

        if effect:
            self.car_body.angle += steer_angle * self.simstep
        self.car_body.angular_velocity = 0


        driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
        v = self.car_body.velocity.length
        if (self.car_reverse_driving):
           v = -v
        v += acceleration * self.simstep # acc 30 m/s^2 at most
        if (v > 0):
            self.car_reverse_driving = False
        else:
            #self.car_reverse_driving = True
            v = 0
            
        v = np.clip(v, 0, self.max_speed)
        if effect:
            self.car_body.velocity = v * driving_direction
        
        # quit the game
        for event in pygame.event.get():
            if event.type == QUIT:
                sys.exit(0)
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                sys.exit(0)

        # Update the screen and stuff
        screen.fill(THECOLORS["black"])
        self.space.debug_draw(self.draw_options)
        self.space.step(self.simstep)

        if draw_screen:
            pygame.display.flip()
            
        self.driving_history.append([self.car_body.position, self.car_body.angle, self.car_body.velocity, self.current_goal_id])
        if (len(self.driving_history) > self.max_history_num):
            self.driving_history.pop(0)

        # Get the current location and the readings there.
        x, y = self.car_body.position
        unit_range = 10
        angle_range = math.pi / 3
        readings = self.get_sonar_readings(x, y, self.car_body.angle, unit_range, angle_range, normalize_reading)
        

        # Goal related features
        # Goal direction, 1 channel
        current_goal = self.goals[self.current_goal_id]
        angle_unit = angle_range /unit_range
        goal_direction = self.get_direction(self.car_body.position, self.car_body.angle, current_goal, angle_unit)
        if normalize_reading:
            readings.append(np.clip(abs(goal_direction) / unit_range,-1,1))
        else:
            readings.append(abs(goal_direction))
        section_number = unit_range*2 + 1

        # Whether there's obstacle in the goal direction, 1 channel
        if (goal_direction <= unit_range and goal_direction >= -unit_range):
            if (readings[section_number + goal_direction + unit_range] == 0):
                readings.append(1)
            else:
                readings.append(0)
        else:
            readings.append(0)
            
        # The difference between the distance to the goal of current frame and last frame, 1 channel
        current_goal_dist = (current_goal - self.car_body.position).length
        last_goal = self.goals[self.current_goal_id-1]
        seg_dist = (Vec2d(current_goal[0], current_goal[1]) - Vec2d(last_goal[0], last_goal[1])).length

        readings.append((self.pre_goal_dist - current_goal_dist)/seg_dist)
        self.pre_goal_dist = current_goal_dist


        # Set the reward.
        # Car crashed when any reading == 1
        score = 0
        if self.car_is_crashed(readings):
            score = (self.current_goal_id - current_goal_dist / seg_dist)*100

            self.crashed = True
            readings.append(1)
            if effect:
                if hitting_reaction_mode > 0:
                    if (self.driving_history[0][0] - self.driving_history[-1][0]).length > 1 * MULTI:
                        self.recover_from_crash(self.driving_history[0])
                    else:
                        self.recover_from_crash([self.car_body.init_position, self.car_body.init_angle, (0, 0), 1])
                    self.driving_history.clear()

                else:
                    self.recover_from_crash([self.car_body.init_position, self.car_body.init_angle, (0, 0), 1])
        else:
            readings.append(0)
                      
        reward = self.get_reward(self.W, readings, normalize_reading, reward_type=2)
        state = np.array([readings])

        self.num_steps += 1

        # Check whether the goal need to be updated
        if current_goal_dist < 3 * MULTI:
            self.current_goal_id = self.current_goal_id + 1
            #self.current_goal_id = random.randint(0, len(self.goals)-1)
            if self.current_goal_id >= len(self.goals):
                self.recover_from_crash([self.car_body.init_position, self.car_body.init_angle, (0, 0), 1])
            self.pre_goal_dist = (self.goals[self.current_goal_id] - self.car_body.position).length
        
        if draw_screen:
            #draw current goal
            pygame.draw.circle(screen, (0, 0, 255), (current_goal[0], height - current_goal[1]), 4)
            pygame.display.update()
            clock.tick()
            
        return reward, state, readings, score, v*self.simstep / MULTI

    def move_obstacles(self):
        # Randomly move obstacles around.
        for obstacle in self.obstacles:
            speed = random.randint(1, 5)
            direction = Vec2d(1, 0).rotated(self.car_body.angle + random.randint(-2, 2))
            obstacle.velocity = speed * direction

    def move_cat(self):
        speed = random.randint(20, 200)
        self.cat_body.angle -= random.randint(-1, 1)
        direction = Vec2d(1, 0).rotated(self.cat_body.angle)
        self.cat_body.velocity = speed * direction

    def car_is_crashed(self, readings):
        return self.crashed
        '''
        if readings[0] >= 0.96 or readings[1] >= 0.96 or readings[2] >= 0.96:
            return True
        else:
            return False
        '''
        
    def recover_from_crash(self, states):
        """
        We hit something, so recover.
        """
        
        [position, angle, velocity, goal_id] = states
        self.car_body.position = position
        self.car_body.angle = angle + (np.random.random()*2-1) * math.pi / 8
        self.car_body.velocity = velocity
        self.current_goal_id = goal_id

        '''
        while self.crashed:
            # Go backwards.
            self.car_body.velocity = -100 * driving_direction
            self.crashed = False
            for i in range(10):
                self.car_body.angle += .2  # Turn a little.
                #screen.fill(THECOLORS["red"])  # Red is scary!'
                self.space.debug_draw(self.draw_options)
                self.space.step(self.simstep)
                if draw_screen:
                    pygame.display.flip()
                clock.tick()
        '''

    # def sum_readings(self, readings):
    #     """Sum the number of non-zero readings."""
    #     tot = 0
    #     print ("readings  :: ", readings)
    #     for i in readings:
    #         tot += i
    #     return tot

    def get_direction(self, base_position, base_direction, goal_position, angle_unit):
        dir = goal_position - base_position
        angle = math.atan2(dir[1], dir[0])
        relative_angle = angle - base_direction
        if relative_angle > math.pi:
            relative_angle -= math.pi
        elif relative_angle < -math.pi:
            relative_angle += math.pi

        section_id = round(relative_angle / angle_unit)
        return section_id

    def get_sonar_readings(self, x, y, angle, unit_range = 10, angle_range = math.pi / 3, normalize_reading = False):
        readings = []
        """
        Instead of using a grid of boolean(ish) sensors, sonar readings
        simply return N "distance" readings, one for each sonar
        we're simulating. The distance is a count of the first non-zero
        reading starting at the object. For instance, if the fifth sensor
        in a sonar "arm" is non-zero, then that arm returns a distance of 5.
        """
        # Make our arms.
        one_arm, max_dist = self.make_sonar_arm(x, y)
        
        obstacle_types = []
        angle_unit = angle_range /unit_range
        for i in range(-unit_range, unit_range+1):
            one_ray = self.get_arm_distance(one_arm, x, y, angle, i*angle_unit)
            obstacle_types.append(one_ray[1])
            if normalize_reading:
                readings.append(np.clip(one_ray[0] / max_dist, -1, 1))
            else:
                readings.append(one_ray[0])

        max_obs_type = np.max(obstacle_types)
        if max_obs_type == 0:
            max_obs_type = 1
        for obs_type in obstacle_types:
            readings.append(obs_type / max_obs_type) #normalize to 1

        #readings = readings[0:7]
        '''
        obstacleType.append(self.get_arm_distance(arm_left, x, y, angle, angle_unit)[1])
        obstacleType.append(self.get_arm_distance(arm_middle, x, y, angle, 0)[1])
        obstacleType.append(self.get_arm_distance(arm_right, x, y, angle, -angle_unit)[1])
        ObstacleNumber = np.zeros(self.num_obstacles_type)

        for i in obstacleType:
            if i == 0: # #sensors seeing black color ( /3 to normalize)
                ObstacleNumber[0] += 1
            elif i == 1: # #sensors seeing yellow color ( /3 to normalize)
                ObstacleNumber[1] += 1
            elif i == 2: # #sensors seeing brown color ( /3 to normalize)
                ObstacleNumber[2] += 1
            elif i == 3: # #sensors seeing red color ( /3 to normalize)
                ObstacleNumber[3] += 1

        # Rotate them and get readings.
        readings.append(1.0 - float(self.get_arm_distance(arm_left, x, y, angle, angle_unit)[0]/39.0)) # 39 = max distance
        readings.append(1.0 - float(self.get_arm_distance(arm_middle, x, y, angle, 0)[0]/39.0))
        readings.append(1.0 - float(self.get_arm_distance(arm_right, x, y, angle, -angle_unit)[0]/39.0))
        
        '''

        if show_sensors:
            pygame.display.update()
        
        return readings

    def get_arm_distance(self, arm, x, y, angle, offset):
        # Used to count the distance.
        i = 0

        # Look at each point and see if we've hit something.
        for point in arm:
            i += 1

            # Move the point to the right spot.
            rotated_p = self.get_rotated_point(
                x, y, point[0], point[1], angle + offset
            )

            # Check if we've hit something. Return the current i (distance)
            # if we did.
            if rotated_p[0] <= 0 or rotated_p[1] <= 0 \
                    or rotated_p[0] >= width or rotated_p[1] >= height:
                return [i, 3]  # Sensor is off the screen, return 3 for wall obstacle
            else:
                obs = screen.get_at(rotated_p)
                temp = self.get_track_or_not(obs)
                if temp != 0:
                    return [i, temp] #sensor hit a round obstacle, return the type of obstacle

            if show_sensors:
                pygame.draw.circle(screen, (255, 255, 255), (rotated_p), 1)
                #pygame.display.update()
                #clock.tick()


        # Return the distance for the arm.
        return [i, 0] #sensor did not hit anything return 0 for black space

    def make_sonar_arm(self, x, y, distance=12, spread=4, point_num=100):
        #spread: Spread between two adjacent points.
        #distance: Gap before first sensor.
        arm_points = []
        # Make an arm. We build it flat because we'll rotate it about the
        # center later.
        for i in range(point_num):
            arm_points.append((distance + x + (spread * i), y))
            
        max_dist = point_num #distance + spread * (point_num-1)
        return arm_points, max_dist

    def get_rotated_point(self, x_1, y_1, x_2, y_2, radians):
        # Rotate x_2, y_2 around x_1, y_1 by angle.
        x_change = (x_2 - x_1) * math.cos(radians) + \
            (y_2 - y_1) * math.sin(radians)
        y_change = (y_1 - y_2) * math.cos(radians) - \
            (x_1 - x_2) * math.sin(radians)
        new_x = x_change + x_1
        new_y = height - (y_change + y_1)
        return int(new_x), int(new_y)

    # def get_track_or_not(self, reading):
    #     if reading == THECOLORS['black']:
    #         return 0
    #     else:
    #         return 1

    def get_track_or_not(self, reading): # basically differentiate b/w the objects the car views.
        if reading == THECOLORS['yellow']:
            return 2  # Sensor is on a yellow obstacle
        elif reading == THECOLORS['brown']:
            return 1  # Sensor is on brown obstacle
        else:
            return 0 #for black


if __name__ == "__main__":
    weights = [1, 1, 1, 1, 1, 1, 1, 1]
    game_state = GameState(weights)
    while True:
        game_state.frame_step((random.randint(0, 2)))

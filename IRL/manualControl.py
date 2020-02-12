"""
Manually control the agent to provide expert trajectories.
The main aim is to get the feature expectaitons of the expert trajectories.

Left arrow key: turn Left
right arrow key: turn right
up arrow key: dont turn, move forward

If curses is used:
down arrow key: exit 
Always exit using down arrow key rather than Ctrl+C or your terminal will be tken over by curses
"""

from simulation import carmunk
import random
from random import randint
import numpy as np
from neuralNets import net1
import msvcrt
import math
#import curses # for keypress, doesn't work for Windows


NUM_FEATURES = 47
GAMMA = 0.9 # the discount factor for RL algorithm

def demo(screen):
    while True:
        screen.print_at('Hello world!',
                        randint(0, screen.width), randint(0, screen.height),
                        colour=randint(0, screen.colours - 1),
                        bg=randint(0, screen.colours - 1))
        ev = screen.get_key()
        if ev in (ord('Q'), ord('q')):
            return
        screen.refresh()
    
def play(use_expert = False):
    '''
    The goal is to get feature expectations of a policy. Note that feature expectations are independent from weights.
    '''
    
    game_state = carmunk.GameState(scene_file_name = 'scenes/scene-city.txt', use_expert=True)   # set up the simulation environment
    game_state.frame_step((11)) # make a forward move in the simulation
    currFeatureExp = np.zeros(NUM_FEATURES)
    prevFeatureExp = np.zeros(NUM_FEATURES)

    moveCount = 0
    carPos = [80, 0]
    carVelo = [0,0]
    carAngle = math.pi / 2
    while True:
        moveCount += 1
        carPos[1] += 0.1

        if (use_expert):
            action = game_state.get_expert_action()
            #action = game_state.get_expert_action_out(carPos, carVelo, carAngle)
        else:
            # get the actual move from keyboard
            move = msvcrt.getch()
            if move == b'H':   # UP key -- move forward
                #action = 2
                action = 14
            elif move == b'K': # LEFT key -- turn left
                #action = 1
                action = 20
            elif move == b'M': # RIGHT key -- turn right
                #action = 0
                action = 0
            else:
                #action = 2
                action = random.randrange(25)
            #print(action)

        '''
        # curses
        event = screen.getch()
        if event == curses.KEY_LEFT:
            action = 1
        elif event == curses.KEY_RIGHT:
            action = 0
        elif event == curses.KEY_DOWN:
            break
        else:
            action = 2
        '''

        # take an action 
        # start recording feature expectations only after 100 frames
        _, _, readings, _ = game_state.frame_step(action)

        if moveCount > 100:
            currFeatureExp += (GAMMA**(moveCount-101))*np.array(readings)
            
        # report the change percentage
        changePercentage = (np.linalg.norm(currFeatureExp - prevFeatureExp)*100.0)/np.linalg.norm(currFeatureExp)

        #print (moveCount)
        #print ("The change percentage in feature expectation is: ", changePercentage)
        prevFeatureExp = np.array(currFeatureExp)

        if moveCount % 20000 == 0:
            break

    return currFeatureExp

if __name__ == "__main__":
    '''
    # Use curses to generate keyboard input
    screen = curses.initscr()
    curses.noecho()
    curses.curs_set(0)
    screen.keypad(1)
    screen.addstr("Play the game")
    curses.endwin()
    '''
    use_expert = True
    featureExp = play(use_expert)
    print('[', end='')
    for feature in featureExp:
        print("%.5f" % round(feature,5), end=', ')
    print (']')

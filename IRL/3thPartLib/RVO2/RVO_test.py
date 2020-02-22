from ctypes import *
import math
import ctypes

#RVO_warper = windll.RVO_warper
RVO_warper = ctypes.WinDLL ("RVO2/RVO_warper.dll")

goals = []

class Vector2(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float)]


def setupScenario(RVO_handler):
    RVO_warper.RVO_setTimeStep(RVO_handler, c_float(0.25))
    
    velocity = Vector2(0, 0)
    # Specify the default parameters for agents that are subsequently added.
    RVO_warper.RVO_setAgentDefaults(RVO_handler, c_float(15.0), 10, c_float(10.0), c_float(10.0), c_float(1.5), c_float(2.0), velocity)

    # Add agents, specifying their start position, and store their goals on the
    # opposite side of the environment.
    for i in range(250):
        position = Vector2(200.0 * math.cos(i * 2.0 * math.pi / 250.0), 200.0 * math.sin(i * 2.0 * math.pi / 250.0))
        RVO_warper.RVO_addAgent(RVO_handler, position)
        goals.append(Vector2(-position.x, -position.y))


def setPreferredVelocities(RVO_handler):
    # Set the preferred velocity to be a vector of unit magnitude (speed) in the
    # direction of the goal.
    agentNum = RVO_warper.RVO_getNumAgents(RVO_handler)

    for i in range(agentNum):
        position = Vector2(0, 0)
        RVO_warper.RVO_getAgentPosition(RVO_handler, i, byref(position))
        goalVector = Vector2(goals[i].x - position.x, goals[i].y - position.y)

        norm = math.sqrt(goalVector.x*goalVector.x + goalVector.y*goalVector.y)
        if (norm > 1.0):
            goalVector.x = goalVector.x / norm
            goalVector.y = goalVector.y / norm

        RVO_warper.RVO_setAgentPrefVelocity(RVO_handler, i, goalVector)


def reachedGoal(RVO_handler):
    # Check if all agents have reached their goals.
    agentNum = RVO_warper.RVO_getNumAgents(RVO_handler)
    for i in range(agentNum):
        position = Vector2(0, 0)
        RVO_warper.RVO_getAgentPosition(RVO_handler, i, byref(position))
        radius = c_float(0)
        RVO_warper.RVO_getAgentRadius(RVO_handler, i, byref(radius))
        deltaVector = Vector2(position.x - goals[i].x, position.y - goals[i].y)
        if (deltaVector.x*deltaVector.x + deltaVector.y*deltaVector.y > radius.value * radius.value):
            return False

    return True


if __name__== "__main__":
    RVO_handler = c_longlong(0)
    RVO_warper.RVO_createEnvironment(byref(RVO_handler))

    # Set up the scenario.
    setupScenario(RVO_handler);

    # Perform (and manipulate) the simulation.
    while reachedGoal(RVO_handler) == False:
        setPreferredVelocities(RVO_handler)
        RVO_warper.RVO_doStep(RVO_handler)

    print("Arrived!");

    RVO_warper.RVO_deleteEnvironment(byref(RVO_handler))

from grid_map_env.classes.action import Action
from grid_map_env.utils import *

import random


class Policy:
    def __init__(self) -> None:
        pass

    def get_action(self, house_map, robot_state):
        ''' 
        Calculate a legal action.
        Here we demonstrate a very simple policy that 
        does not perform any from of search.
        Parameters:
            house_map (list): A 2D list representing the house map.
            robot_state (RobotState): The current state of the robot.
        Returns: 
            action (Action): Action for execution
        '''

        if robot_state.speed < 2:
            acc = 1  # accelerate
        else:
            acc = -1 # decelerate

        action = Action(acc=acc, rot=0)  
        
        next_state = transition(robot_state=robot_state, action=action)

        if is_collision(house_map=house_map, robot_state=next_state):
            if robot_state.speed > 0:
                action = Action(acc=-1, rot=0)
            else:
                random_action = random.choice([(0, 1), (0, -1)])
                action = Action(acc=random_action[0], rot=random_action[1])

        return action  # return the action for execution
    
def transition(robot_state, action):

    temp_state = robot_state.copy()
    temp_state.speed += action.acc
    temp_state.speed = max(min(temp_state.speed, 3), 0)


    if temp_state.speed != 0:
        if temp_state.direction == 0:
            temp_state.col -= temp_state.speed
        elif temp_state.direction == 1:
            temp_state.row -= temp_state.speed
        elif temp_state.direction == 2:
            temp_state.col += temp_state.speed
        elif temp_state.direction == 3:
            temp_state.row += temp_state.speed

    else:
        temp_state.direction = (temp_state.direction+action.rot) % 4   
        
    return temp_state 


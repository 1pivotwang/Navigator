from grid_map_env.classes.action import Action
from grid_map_env.utils import *
import numpy as np
import random


class Policy:
    #members:Q(s,a) 9 actions,100*100*4*4 states(4 directions,4 speeds),100*100 space states;some are not valid
    Q=np.zeros((100,100,4,4,9)) 
    N=np.zeros((100,100,4,4,9)) #to record visited states
    T=[] #tree nodes
    pi_0=np.zeros((100,100,4,4)) #initial policy(rollout policy),can be renewed
    def __init__(self) -> None:
        pass
    
    def acceptor(self,start_state,goal_state): #专门用于传入一些必要的东西
        self.start_state=start_state
        self.goal_state=goal_state
        self.T.append((start_state.row,start_state.col,start_state.direction,start_state.speed)) #根节点不进行rollout??

    def action_encoder(self,acc,rot): #0-8的数可以写为除以3的商和余数，我们以此来编码
        return (acc+1)*3+(rot+1)
    def action_decoder(self,action):
        return (action//3-1,action%3-1)
    
    def action_available(self,robot_state):
        if robot_state.speed==0:
            return [self.action_encoder(0,0),self.action_encoder(1,0),self.action_encoder(0,1),self.action_encoder(0,-1)]
        elif robot_state.speed==3:
            return [self.action_encoder(0,0),self.action_encoder(-1,0)]
        else:
            return [self.action_encoder(0,0),self.action_encoder(-1,0),self.action_encoder(1,0)]
    
    
    def policy_initialization(self,house_map): #原计划是能动态更新的policy，先选一个有道理的初始policy，就算不能动态更新，也可以作为rollout policy的prefer情况（rollout并不是全随机，概率不等），是否要更新概率？？
        for i in range(100):
            for j in range(100):
                delta_x,delta_y=self.goal_state.row-i,self.goal_state.col-j
                for k in range(4):
                    for l in range(4):
                        if is_collision(house_map=house_map,robot_state=RobotState(row=i,col=j,direction=k,speed=l)):
                            continue
                        else:
                            #依照直觉来设置，把方向调整到向着终点
                            #保守起见，速度控制到1
                            if l>1:
                                self.pi_0[i,j,k,l]=self.action_encoder(-1,0)
                            elif l==1:
                                if abs(delta_x)>abs(delta_y): #x方向更远,向x方向移动
                                    if delta_x>0:
                                        if k==2:
                                            self.pi_0[i,j,k,l]=self.action_encoder(0,0)
                                        else:
                                            self.pi_0[i,j,k,l]=self.action_encoder(-1,0)
                                    else:
                                        if k==0:
                                            self.pi_0[i,j,k,l]=self.action_encoder(0,0)
                                        else:
                                            self.pi_0[i,j,k,l]=self.action_encoder(-1,0)
                                else: #y方向更远
                                    if delta_y>0:
                                        if k==1:
                                            self.pi_0[i,j,k,l]=self.action_encoder(0,0)
                                        else:
                                            self.pi_0[i,j,k,l]=self.action_encoder(-1,0)
                                    else:
                                        if k==3:
                                            self.pi_0[i,j,k,l]=self.action_encoder(0,0)
                                        else:
                                            self.pi_0[i,j,k,l]=self.action_encoder(-1,0)
                            else:
                                if abs(delta_x)>abs(delta_y):
                                    if delta_x>0:
                                        if k==2:
                                            self.pi_0[i,j,k,l]=self.action_encoder(1,0)
                                        else:
                                            if k==1:
                                                self.pi_0[i,j,k,l]=self.action_encoder(0,1)
                                            else:
                                                self.pi_0[i,j,k,l]=self.action_encoder(0,-1)
                                    else:
                                        if k==0:
                                            self.pi_0[i,j,k,l]=self.action_encoder(1,0)
                                        else:
                                            if k==1:
                                                self.pi_0[i,j,k,l]=self.action_encoder(0,-1)
                                            else:
                                                self.pi_0[i,j,k,l]=self.action_encoder(0,1)
                                else:
                                    if delta_y>0:
                                        if k==1:
                                            self.pi_0[i,j,k,l]=self.action_encoder(1,0)
                                        else:
                                            if k==2:
                                                self.pi_0[i,j,k,l]=self.action_encoder(0,-1)
                                            else:
                                                self.pi_0[i,j,k,l]=self.action_encoder(0,1)
                                    else:
                                        if k==3:
                                            self.pi_0[i,j,k,l]=self.action_encoder(1,0)
                                        else:
                                            if k==2:
                                                self.pi_0[i,j,k,l]=self.action_encoder(0,1)
                                            else:
                                                self.pi_0[i,j,k,l]=self.action_encoder(0,-1)


    #这里先实现老师给的伪代码,reward未定
    def simulate(self,robot_state,d): #d is the remaining steps for one rollout,第一个想法，不采取离散的reward，直接采取一整次rollout的reward，第二个想法，依照老师的代码，但是拟合出reward，利用差分计算出单步reward
        if d==0:
            return 0
        robot_state_tupple=(robot_state.row,robot_state.col,robot_state.direction,robot_state.speed)
        if robot_state_tupple not in self.T:
            self.T.append(robot_state_tupple)
            return self.rollout(robot_state,d)
        action_available=self.action_available(robot_state)
        N_s_a=[self.N[robot_state_tupple[0],robot_state_tupple[1],robot_state_tupple[2],robot_state_tupple[3],action] for action in action_available] 
        N_s=np.sum(N_s_a)
        intermediate=lambda q,n,n_s_a,c:q+c*np.sqrt(np.log(n)/n_s_a)
        Q_s_a=np.array([])
        for n_s_a in N_s_a:
            if n_s_a==0:
                Q_s_a.append(np.inf)
            else:
                Q_s_a.append(intermediate(self.Q[robot_state_tupple[0],robot_state_tupple[1],robot_state_tupple[2],robot_state_tupple[3],iter(action_available).__next__()],N_s,n_s_a,2))
        a=np.argmax(Q_s_a)

        next_state=self.transition(robot_state,self.action_decoder(action_available[a]))
        #出现分歧，涉及到reward的问题，还有collision的判断问题，collision不应该包在transition里面吗？？？
        
        
    def rollout(self,robot_state,d): #d is the remaining steps for one rollout
        if d==0:
            return 0
        #出现分歧的时候，随机选择一个action还是有策略，以概率表达策略？？？
        


    def get_action(self, house_map, robot_state):
        ''' 
        Calculate a legal action.
        Here we demonstrate a very simple policy that 
        does not perform any from of search.
        Args: 
            house map (list of list): a 2D list of integers representing the house map. Please refer to Table 6 for its encoding.
            robot state (RobotState): an instance of the RobotState class representing the current state of the robot.
        Returns:
             action (Action): an instance of Action class representing the action for execution.
        '''

        """
        if robot_state.speed < 2:
            acc = 1  # accelerate
        else:
            acc = -1 # decelerate

        action = Action(acc=acc, rot=0)  # construct an instance of the Action class
        
        next_state = self.transition(robot_state=robot_state, action=action) # predict the transition

        # collision checking and response
        if is_collision(house_map=house_map, robot_state=next_state):
            #change the action due to collision in the predicted enxt state
            if robot_state.speed > 0: # decelerate to stop
                action = Action(acc=-1, rot=0)
            else: # choose random action
                random_action = random.choice([(0, 1), (0, -1)])
                action = Action(acc=random_action[0], rot=random_action[1])
        """
        
        return action  # return the action for execution
    
    def transition(self,robot_state, action): #这个不会导致位置溢出吗，它并不检查是否碰撞啊，输出的结果只是理论情况）
        '''
        a simple example for transition function
        Args:
            robot state (RobotState): an instance of the RobotState class representing the current state of the robot.
            action (Action): an instance of Action class representing the action for execution.
        Returns:
            next state (RobotState): an instance of the RobotState class representing the predicted state of the robot.
        '''

        next_state = robot_state.copy() #deep copy the robot state

        # update the robot's speed
        next_state.speed += action.acc 
        next_state.speed = max(min(next_state.speed, 3), 0)

        #update the robot's position
        if next_state.speed != 0:
            if next_state.direction == 0:
                next_state.col -= next_state.speed
            elif next_state.direction == 1:
                next_state.row -= next_state.speed
            elif next_state.direction == 2:
                next_state.col += next_state.speed
            elif next_state.direction == 3:
                next_state.row += next_state.speed

        #update the robot's direction
        else:
            next_state.direction = (next_state.direction+action.rot) % 4   
            
        return next_state 


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
    Transition=np.zeros((4,4,9,4,4)) #从一个（速度，方向）状态转移到另一个（速度，方向）状态的概率
    initialization=1 #用于只用一次的启动


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
    def simulate(self,house_map,robot_state,d): #d is the remaining steps for one rollout,第一个想法，不采取离散的reward，直接采取一整次rollout的reward，第二个想法，依照老师的代码，但是拟合出reward，利用差分计算出单步reward
        gamma=0.9

        if d==0:
            return 0
        robot_state_tupple=(robot_state.row,robot_state.col,robot_state.direction,robot_state.speed)
        if robot_state_tupple not in self.T:
            self.T.append(robot_state_tupple)
            return self.rollout(house_map,robot_state,d)
        action_available=self.action_available(robot_state)
        N_s_a=[self.N[robot_state_tupple[0],robot_state_tupple[1],robot_state_tupple[2],robot_state_tupple[3],action] for action in action_available] 
        N_s=np.sum(N_s_a)
        intermediate=lambda q,n,n_s_a,c:q+c*np.sqrt(np.log(n)/n_s_a)
        Q_s_a=[]
        for n_s_a in N_s_a:
            if n_s_a==0:
                Q_s_a.append(np.inf)
            else:
                Q_s_a.append(intermediate(self.Q[robot_state_tupple[0],robot_state_tupple[1],robot_state_tupple[2],robot_state_tupple[3],iter(action_available).__next__()],N_s,n_s_a,2))
        a=np.argmax(Q_s_a)
        #采样一个sample
        next_state,error,collision=self.transition(house_map,robot_state,action_available[a])
        q=self.reward_function(house_map,robot_state,action_available[a],next_state,error,collision)+gamma*self.simulate(house_map,next_state,d-1)
        self.N[robot_state_tupple[0],robot_state_tupple[1],robot_state_tupple[2],robot_state_tupple[3],action_available[a]]+=1
        QQ=self.Q[robot_state_tupple[0],robot_state_tupple[1],robot_state_tupple[2],robot_state_tupple[3],action_available[a]]
        NN=self.N[robot_state_tupple[0],robot_state_tupple[1],robot_state_tupple[2],robot_state_tupple[3],action_available[a]]
        self.Q[robot_state_tupple[0],robot_state_tupple[1],robot_state_tupple[2],robot_state_tupple[3],action_available[a]]+=(q-QQ)/NN
        return q
        
        
    def rollout(self,house_map,robot_state,d): #d is the remaining steps for one rollout, 没有transition函数，爆栈，碰撞
        if d==0:
            return 0
        #根据rollout policy采样一个action,非常值得修正
        action=self.pi_0[int(robot_state.row),int(robot_state.col),int(robot_state.direction),int(robot_state.speed)]
        next_state,error,collision=self.transition(house_map,robot_state,action)
        return self.reward_function(house_map,robot_state,action,next_state,error,collision)+self.rollout(house_map,next_state,d-1)
        
        
    def reward_function(self,house_map,robot_state,action,next_robot_state,error,collision): #error是指他有没有不按照指令前进,collision是指过程中是否发生了碰撞
        #到达终点则有超级大reward
        reward=0.0
        if self.goal_state.row==next_robot_state.row and self.goal_state.col==next_robot_state.col and next_robot_state.speed==0:
            return 300.0
        #先改速度，和方向，再动 
        #计算一下Manhattan距离
        next_delta_x,next_delta_y=self.goal_state.row-next_robot_state.row,self.goal_state.col-next_robot_state.col
        next_Manhattan_distance=abs(next_delta_x)+abs(next_delta_y)
        current_delta_x,current_delta_y=self.goal_state.row-robot_state.row,self.goal_state.col-robot_state.col
        current_Manhattan_distance=abs(current_delta_x)+abs(current_delta_y)
        #减小manhattan距离将会有reward
        if next_Manhattan_distance<current_Manhattan_distance:
            reward+=10.0*current_Manhattan_distance/(current_Manhattan_distance-next_Manhattan_distance) #这保障了不会出现除0的情况,分母也不是0，30为超参数
            #在减小的基础上，如果加速，应该reward更大一点
            reward+=10.0*abs(next_robot_state.speed-robot_state.speed)
        #改变方向的reward或者速度的reward(走了一步所以给点reward)，我考虑给一个随机的reward，以克服我不知道reward的问题
        reward+=abs(20.0*random.random())
        #暂时到这里,可以接着完善



        #截断reward
        reward=min(reward,50)
        #如果他并不是按照指令前进的，那么reward应该打折扣，以概率为折扣依据
        if error==1:
            alpha=0.2
            reward*=alpha
        #如果他碰撞了，那么reward更应小
        if collision==1:
            reward*=0.1
        return reward
        


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
        for i in range(100):
            self.simulate(house_map,robot_state,50)
        action_available=self.action_available(robot_state)
        Q_s_a=[self.Q[robot_state.row,robot_state.col,robot_state.direction,robot_state.speed,action] for action in action_available]
        a=np.argmax(Q_s_a)
        action=self.action_decoder(action_available[a])
        return Action(acc=action[0],rot=action[1])  #return the action for execution
    
    def transition(self,house_map,robot_state, action): #尽可能模拟出来的step函数
        '''
        a simple example for transition function
        Args:
            robot state (RobotState): an instance of the RobotState class representing the current state of the robot.
            action (Action): an instance of Action class representing the action for execution.
        Returns:
            next state (RobotState): an instance of the RobotState class representing the predicted state of the robot.
        '''
        '''
        next_state = robot_state.copy() #deep copy the robot state
        action_decoded=self.action_decoder(action)
        # update the robot's speed
        next_state.speed += action_decoded[0]
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
            next_state.direction = (next_state.direction+action_decoded[1]) % 4   
        
        error=1
        collision=0
        '''
        #现在考虑的是与位置无关，只考虑速度和方向，然后进行了一个模拟
        #执行模拟，先更新速度，方向
        #（先取出每一种情况的概率，如果太小了就直接忽略）
        #一共16种情况，采用同样的商和余数的编码模式
        encoder=lambda v,d:v*4+d+1
        decoder=lambda n:(n//4,n%4)
        probabilities=[self.Transition[robot_state.speed,robot_state.direction,action,decoder(i)[0],decoder(i)[1]] for i in range(16)]
        next_encoded_v_d=random.choice(range(16),p=probabilities) #概率生成随机状态
        next_v_d=decoder(next_encoded_v_d)
        intermediate_state=RobotState(robot_state.row,robot_state.col,direction=next_v_d[1],speed=next_v_d[0])
        former_intermediate_state=intermediate_state.copy() #用于回溯
        #出发，真开始走，首先要假定原本的是合法的位置
        #逐步检测碰撞
        #机器人碰撞逻辑为当前速度减一的位置，但是这个速度是虚拟速度，可以是4，方向不变，速度为0
        collision=0
        error=0
        #检测是否发生了错误
        decoded_action=self.action_decoder(action)
        ideal_state=robot_state.copy()
        ideal_state.speed+=decoded_action[0]
        ideal_state.speed=max(min(ideal_state.speed,3),0)
        ideal_state.direction=(ideal_state.direction+decoded_action[1])%4 #指令本身是合法的，更新速度与方向可得到不出错的理想状态
        if ideal_state.speed==intermediate_state.speed and ideal_state.direction==intermediate_state.direction:
            error=0
        else:
            error=1

        #检测是否发生了碰撞
        if intermediate_state.speed==0:
            #不可能碰撞
            pass
        else:
            step_counter=0
            while step_counter<intermediate_state.speed:
                step_counter+=1

                if intermediate_state.direction==0:
                    intermediate_state.col-=1
                elif intermediate_state.direction==1:
                    intermediate_state.row-=1
                elif intermediate_state.direction==2:
                    intermediate_state.col+=1
                else:
                    intermediate_state.row+=1
                
                if is_collision(house_map,intermediate_state):
                    collision=1
                    #说明上一态就撞了，计算出撞后的情况，依照速度反向走一段
                    virtual_speed=intermediate_state.speed
                    if(intermediate_state.speed==3 and robot_state.speed==3 and self.action_decoder(action)[0]==1): #很重要
                        virtual_speed=4
                    if former_intermediate_state.direction==0:
                        former_intermediate_state.col+=virtual_speed-1
                    elif intermediate_state.direction==1:
                        former_intermediate_state.row+=virtual_speed-1
                    elif intermediate_state.direction==2:
                        former_intermediate_state.col-=virtual_speed-1
                    else:
                        former_intermediate_state.row-=virtual_speed-1
                    intermediate_state=former_intermediate_state.copy() #回溯加撞的处理
                    break
        
        next_state=intermediate_state.copy()
        return (next_state,error,collision)
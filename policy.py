from grid_map_env.classes.action import Action
from grid_map_env.utils import *
import numpy as np
import random
import datetime
import heapq   #优先队列
#分解问题设计中间goal state？？？？

class CustomPriorityQueue():
    def __init__(self):
        self.queue = []
    def put(self, item):
        heapq.heappush(self.queue, item)
    def get(self):
        return heapq.heappop(self.queue)
    def remove(self,item):
        self.queue=[(p,i) for p,i in self.queue if i!=item]
        heapq.heapify(self.queue)
    def empty(self):
        return len(self.queue)==0
    
class Policy:
    open_set=CustomPriorityQueue()
    real_open_set=CustomPriorityQueue()
    #members:Q(s,a) 9 actions,100*100*4*4 states(4 directions,4 speeds),100*100 space states;some are not valid
    house_map=None
    Q=np.zeros((100,100,4,4,9)) 
    N=np.zeros((100,100,4,4,9)) #to record visited states
    T=[] #tree nodes
    #pi_0=np.zeros((100,100,4,4)) #initial policy(rollout policy),can be renewed
    Transition=np.zeros((4,4,9,4,4)) #从一个（速度，方向）状态转移到另一个（速度，方向）状态的概率
    initialization=1 #用于只用一次的启动
    steps_counter=0 #用于计算已经走的步数
    action_list=[]
    path=None #用于存储A*算法的算出的路径
    real_path=None #用于存储A*算法的算出的真正路径
    current_position_in_path=None #用于机器人在路径中的位置
    current_goal_in_path=None #用于机器人在路径中的下几个目标
    goal_num=5 #选取最多四个目标,重要参数

    def __init__(self) -> None:
        pass
    
    def acceptor(self,start_state,goal_state): #专门用于传入一些必要的东西
        self.start_state=start_state
        self.goal_state=goal_state
        

    def action_encoder(self,acc,rot): #0-8的数可以写为除以3的商和余数，我们以此来编码
        return (acc+1)*3+(rot+1)
    def action_decoder(self,action):
        return (action//3-1,action%3-1)
    
    def state_encoder(self,robot_state): #高维度数组展开成一维度
        return int(robot_state.row*100+robot_state.col)
    
    def state_decoder(self,state):
        s=int(state)
        return (s//100,s%100)
    
    def real_state_encoder(self,robot_state):
        return int(robot_state.row*1600+robot_state.col*16+robot_state.direction*4+robot_state.speed)
    
    def real_state_decoder(self,state):
        s=int(state)
        return (s//1600,s%1600//16,s%1600%16//4,s%1600%16%4)
    
    def path_mender(self,house_map,robot_state):
        #根据现有的信息修正path,主要靠枚举
        #记录现在的前进方向
        turning_point_one=0
        turning_point_two=None
        turning_point_three=None
        turning_point_four=None
        direction_one=None
        direction_two=None
        for i in range(turning_point_one,len(self.path)):
            if turning_point_one==i:
                continue
            if(direction_one==None):
                direction_one=tuple(a-b for a,b in zip(self.state_decoder(self.path[i]),self.state_decoder(self.path[i-1])))
                continue
            else:
                direction_two=tuple(a-b for a,b in zip(self.state_decoder(self.path[i]),self.state_decoder(self.path[i-1])))
                #判断是否出现直角
                if direction_one[0]*direction_two[0]+direction_one[1]*direction_two[1]==0:
                    if(turning_point_two==None):
                        turning_point_two=i-1
                        direction_one=direction_two
                        direction_two=None
                        continue
                    if(turning_point_three==None):
                        #计算点积
                        vec1=tuple(a-b for a,b in zip(self.state_decoder(self.path[turning_point_two]),self.state_decoder(self.path[turning_point_one])))
                        vec2=tuple(a-b for a,b in zip(self.state_decoder(self.path[i]),self.state_decoder(self.path[i-1])))
                        dot_product=vec1[0]*vec2[0]+vec1[1]*vec2[1]
                        if dot_product>0:
                            turning_point_three=i-1
                            direction_one=tuple(a-b for a,b in zip(self.state_decoder(self.path[i]),self.state_decoder(self.path[i-1])))
                            direction_two=None
                            continue
                        else:
                            #从头再来
                            turning_point_one=i
                            turning_point_two=None
                            turning_point_three=None
                            turning_point_four=None
                            direction_one=None
                            direction_two=None
                            continue
                    if(turning_point_four==None):
                        #计算点积
                        vec1=tuple(a-b for a,b in zip(self.state_decoder(self.path[turning_point_three]),self.state_decoder(self.path[turning_point_two])))
                        vec2=tuple(a-b for a,b in zip(self.state_decoder(self.path[i]),self.state_decoder(self.path[i-1])))
                        dot_product=vec1[0]*vec2[0]+vec1[1]*vec2[1]
                        if dot_product>0: #出现阶梯，也许可以优化
                            #优化
                            turning_point_four=i-1
                            #检查有没有障碍
                            possible=1
                            point_two=self.state_decoder(self.path[turning_point_two])
                            point_three=self.state_decoder(self.path[turning_point_three])
                            point_four=self.state_decoder(self.path[turning_point_four])
                            if point_two[0]<point_four[0]:
                                for row in range(point_two[0],point_four[0]+1):
                                    if is_collision(house_map,RobotState(row=row,col=point_two[1],direction=0,speed=0)) or is_collision(house_map,RobotState(row=row,col=point_four[1],direction=0,speed=0)):
                                        possible=0
                                        break
                            else:
                                for row in range(point_four[0],point_two[0]+1):
                                    if is_collision(house_map,RobotState(row=row,col=point_two[1],direction=0,speed=0)) or is_collision(house_map,RobotState(row=row,col=point_four[1],direction=0,speed=0)):
                                        possible=0
                                        break 
                            
                            if point_two[1]<point_four[1]:
                                for col in range(point_two[1],point_four[1]+1):
                                    if is_collision(house_map,RobotState(row=point_two[0],col=col,direction=0,speed=0)) or is_collision(house_map,RobotState(row=point_four[0],col=col,direction=0,speed=0)):
                                        possible=0
                                        break
                            else:
                                for col in range(point_four[1],point_two[1]+1):
                                    if is_collision(house_map,RobotState(row=point_two[0],col=col,direction=0,speed=0)) or is_collision(house_map,RobotState(row=point_four[0],col=col,direction=0,speed=0)):
                                        possible=0
                                        break
                            if possible==0:
                                #从头再来
                                turning_point_one=i
                                turning_point_two=None
                                turning_point_three=None
                                turning_point_four=None
                                direction_one=None
                                direction_two=None
                                continue
                            else:
                                #优化
                                slice_one=self.path[0:turning_point_two+1]
                                slice_two=self.path[turning_point_two+1:turning_point_three+1]
                                slice_three=self.path[turning_point_three+1:turning_point_four+1]
                                slice_four=self.path[turning_point_four+1:]
                                #重新组合,按照one，three，two，four的顺序
                                self.path=slice_one+slice_three+slice_two+slice_four
                                turning_point_one=i-1
                                turning_point_two=None
                                turning_point_three=None
                                turning_point_four=None
                                direction_one=None
                                direction_two=None
                                continue
                        else: #从头再来
                            turning_point_one=i
                            turning_point_two=None
                            turning_point_three=None
                            turning_point_four=None
                            direction_one=None
                            direction_two=None
                            continue

                        



    def get_goal_in_path(self,house_map,robot_state): #用于获取当前的目标
        #首先是更新current_position_in_path
        radius=8
        """
        while True:
            current_forward_vector=None
            current_backward_vector=None
            if self.current_position_in_path==0:
                current_forward_vector=tuple(a-b for a,b in zip(self.state_decoder(self.path[1]),self.state_decoder(self.path[0])))
            elif self.current_goal_in_path==len(self.path)-1:
                current_backward_vector=tuple(a-b for a,b in zip(self.state_decoder(self.path[-1]),self.state_decoder(self.path[-2])))
            else:
                current_forward_vector=tuple(a-b for a,b in zip(self.state_decoder(self.path[self.current_position_in_path+1]),self.state_decoder(self.path[self.current_position_in_path])))
                current_backward_vector=tuple(a-b for a,b in zip(self.state_decoder(self.path[self.current_position_in_path]),self.state_decoder(self.path[self.current_position_in_path-1])))
            #计算robot到当前position的向量
            robot_vector=tuple(a-b for a,b in zip(self.state_decoder(self.path[self.current_position_in_path]),(robot_state.row,robot_state.col)))
            #计算robot到两个向量的投影与向量的比值
            if current_forward_vector!=None:
                forward_projection=sum([a*b for a,b in zip(robot_vector,current_forward_vector)])/(1.0*sum([a**2 for a in current_forward_vector]))
            if current_backward_vector!=None:
                backward_projection=sum([a*b for a,b in zip(robot_vector,current_backward_vector)])/(1.0*sum([a**2 for a in current_backward_vector]))
            #先看是不是要向前走
            if current_forward_vector!=None and forward_projection>0.6:
                self.current_position_in_path+=1
            elif current_backward_vector!=None and backward_projection>0.6:
                self.current_position_in_path-=1
            else:
                break
       
        #依据目前的位置，计算出目标
        if self.current_position_in_path==len(self.path)-1:
            self.current_goal_in_path=[self.current_position_in_path]
        else:
            self.current_goal_in_path=[self.path[self.current_position_in_path+1+i] for i in range(min(self.goal_num,len(self.path)-1-self.current_position_in_path))]
        """
        Euc_distance=[]
        #直接计算到这个点的欧式距离
        for s in self.path:
            s=self.state_decoder(s)
            Euc_distance.append(((s[0]-robot_state.row)**2+(s[1]-robot_state.col)**2)**0.5)
        #找到最近的点
        self.current_position_in_path=np.argmin(Euc_distance)
        #更新current_goal_in_path
        if self.current_position_in_path==len(self.path)-1:
            self.current_goal_in_path=[self.current_position_in_path]
        else:
            f=lambda x,y:((x[0]-y[0])**2+(x[1]-y[1])**2)**0.5
            Euc_distance_to_projection=[f(self.state_decoder(self.path[i]),self.state_decoder(self.path[self.current_position_in_path])) for i in range(self.current_position_in_path+1,len(self.path))]
            for i in range(self.current_position_in_path+1,len(self.path)):
                self.current_goal_in_path=[self.path[i]]
                if Euc_distance_to_projection[i-1-self.current_position_in_path]>radius:
                    self.current_goal_in_path=[self.path[i]]
                    break
            #self.current_goal_in_path=[self.path[self.current_position_in_path+1+i] for i in range(min(self.goal_num,len(self.path)-1-self.current_position_in_path))]


    def get_possible_next_state(self,house_map,current_state):
        decoded_state=self.state_decoder(current_state)
        next_state=[]
        search_range=2 #让他优先搜最远的
        for i in range(1,search_range).__reversed__():
            candidate_state=RobotState(row=decoded_state[0]+i,col=decoded_state[1],direction=0,speed=0)
            if is_collision(house_map,candidate_state):
                continue
            else:
                next_state.append(candidate_state.row*100+candidate_state.col)
        for i in range(1,search_range).__reversed__():
            candidate_state=RobotState(row=decoded_state[0]-i,col=decoded_state[1],direction=0,speed=0)
            if is_collision(house_map,candidate_state):
                continue
            else:
                next_state.append(candidate_state.row*100+candidate_state.col)
        for i in range(1,search_range).__reversed__():
            candidate_state=RobotState(row=decoded_state[0],col=decoded_state[1]+i,direction=0,speed=0)
            if is_collision(house_map,candidate_state):
                continue
            else:
                next_state.append(candidate_state.row*100+candidate_state.col)
        for i in range(1,search_range).__reversed__():
            candidate_state=RobotState(row=decoded_state[0],col=decoded_state[1]-i,direction=0,speed=0)
            if is_collision(house_map,candidate_state):
                continue
            else:
                next_state.append(candidate_state.row*100+candidate_state.col)
        for i in range(len(next_state)):
            next_state[i]=int(next_state[i])
        return next_state
    
    def get_possible_next_real_state(self,house_map,current_state):
        decoded_state=self.real_state_decoder(current_state)
        next_real_state=[]
        actions=self.action_available(RobotState(row=decoded_state[0],col=decoded_state[1],direction=decoded_state[2],speed=decoded_state[3]))
        for action in actions:
            action=self.action_decoder(action)
            #先更新速度，方向
            next_speed=decoded_state[3]+action[0]
            next_speed=max(min(next_speed,3),0)
            next_direction=(decoded_state[2]+action[1])%4
            #再更新位置
            if next_speed!=0:
                if next_direction==0:
                    next_row=decoded_state[0]
                    next_col=decoded_state[1]-next_speed
                elif next_direction==1:
                    next_row=decoded_state[0]-next_speed
                    next_col=decoded_state[1]
                elif next_direction==2:
                    next_row=decoded_state[0]
                    next_col=decoded_state[1]+next_speed
                elif next_direction==3:
                    next_row=decoded_state[0]+next_speed
                    next_col=decoded_state[1]
            else:
                next_row=decoded_state[0]
                next_col=decoded_state[1]
            #判断是否碰撞
            if is_collision(house_map,RobotState(row=next_row,col=next_col,direction=next_direction,speed=next_speed)): #万一必须撞怎么办？？？
                continue
            else:
                next_real_state.append(next_row*1600+next_col*16+next_direction*4+next_speed)
        return next_real_state
    
    def action_available(self,robot_state):
        if robot_state.speed==0:
            return [self.action_encoder(0,0),self.action_encoder(1,0),self.action_encoder(-1,0),self.action_encoder(0,1),self.action_encoder(0,-1)]
        else:
            return [self.action_encoder(0,0),self.action_encoder(-1,0),self.action_encoder(1,0)]
    
    #两个都很简单，也许可以优化
    def heuristic(self,state):
        decoded_state=self.state_decoder(state)
        delta_row=abs(self.goal_state.row-decoded_state[0])
        delta_col=abs(self.goal_state.col-decoded_state[1])
        #优先搜长的边
        if delta_row!=0 and delta_col!=0:
            alpha=2.0*delta_row/(delta_row+delta_col)
            beta=2.0*delta_col/(delta_row+delta_col)
        else:
            alpha=1.0
            beta=1.0
        h=alpha*abs(delta_row)+beta*abs(delta_col)
        current_goal=(self.goal_state.row,self.goal_state.col)
        delta_row=current_goal[0]-decoded_state[0]
        delta_col=current_goal[1]-decoded_state[1]
        
        if self.robot_state.direction==0:
            if delta_col<=0:
                return h
        if self.robot_state.direction==1:
            if delta_row<=0:
                return h
        if self.robot_state.direction==2:
            if delta_col>=0:
                return h
        if self.robot_state.direction==3:
            if delta_row>=0:
                return h
        h+=1.0
        return h 
    def real_heuristic(self,real_state):
        
        decoded_state=self.real_state_decoder(real_state)
        h=abs(decoded_state[0]-self.goal_state.row)/3.0+abs(decoded_state[1]-self.goal_state.col)/3.0
        #对方向进行惩罚
    
        #计算目标的方向
        
        if self.Euclidean_distance_to_final<7:
            current_goal=(self.goal_state.row,self.goal_state.col)
        else:
            current_goal=self.real_state_decoder(self.current_goal_in_path[-1])
        #方向如果不朝着目标，就得加大惩罚
        delta_row=current_goal[0]-decoded_state[0]
        delta_col=current_goal[1]-decoded_state[1]
        
        if self.robot_state.direction==0:
            if delta_col<=0:
                return h
        if self.robot_state.direction==1:
            if delta_row<=0:
                return h
        if self.robot_state.direction==2:
            if delta_col>=0:
                return h
        if self.robot_state.direction==3:
            if delta_row>=0:
                return h
        h+=1.0
        return h
    
    def astar(self,initial_state, goal_state): #这里state是编码后的
        #初始化 open_set
        self.open_set=CustomPriorityQueue()
        self.open_set.put((0,int(initial_state)))
        closed_set = set()
        parents = {}  # Dictionary to store parents
        #inflated A*算法 以加速
        inflation_index=1.8
        step=0
        while not self.open_set.empty():
            current_cost,current_state = self.open_set.get()

            if current_state == goal_state:
                return self.reconstruct_path(parents, goal_state)
            #将open_set中的删除
            self.open_set.remove(current_state)
            
            closed_set.add(int(current_state))
            possible_next_states=self.get_possible_next_state(self.house_map,current_state)
            #调整possible_next_states的顺序
            possible_next_states.sort(key=lambda x:self.heuristic(x))
            for next_state in possible_next_states:

                if next_state in closed_set:
                    continue

                tentative_cost = current_cost + 1
                if next_state not in self.open_set.queue:
                    self.open_set.put( (tentative_cost + inflation_index*self.heuristic(next_state),next_state))
                    parents[next_state] = current_state  # Set parent for backtracking
            #减小inflation_index
            step+=1
            if step%5==0:
                step=0
                if inflation_index>1:
                    inflation_index-=0.1

        return "No path found"
    
    def real_astar(self,initial_state, goal_state):  #这里state是编码后的 goal_state是一个集合
        #初始化 real_open_set
        real_astar_limit=0.99
        self.real_open_set=CustomPriorityQueue()
        self.real_open_set.put((0,int(initial_state)))
        closed_set = set()
        g_score={}
        g_score[initial_state]=0
        parents = {}
        #inflated A*算法 以加速
        step=0
        while not self.real_open_set.empty():
            current_cost,current_state = self.real_open_set.get()
            self.real_astar_now=datetime.datetime.now() #记录时间
            if self.real_astar_now-self.start_time>datetime.timedelta(seconds=real_astar_limit):
                return "No path found"
            for goal in goal_state:
                if current_state == goal:
                    return self.reconstruct_real_path(parents, goal)
            """    
            #将open_set中的删除
            self.real_open_set.remove(current_state)
            
            closed_set.add(int(current_state))
            #问题是这个东西可以为空？？？因为碰撞
            for next_state in self.get_possible_next_real_state(self.house_map,current_state):

                if next_state in closed_set:
                    continue

                tentative_cost = current_cost + 1
                if next_state not in self.real_open_set.queue:
                    self.real_open_set.put( (tentative_cost + self.real_heuristic(next_state),next_state))
                    parents[next_state] = current_state  # Set parent for backtracking
            #减小inflation_index
            """
            for next_state in self.get_possible_next_real_state(self.house_map,current_state):
                self.real_state_decoder(next_state)
                self.real_state_decoder(current_state)
                tentative_g_score=g_score[current_state]+1
                if next_state not in g_score or tentative_g_score<g_score[next_state]:
                    g_score[next_state]=tentative_g_score
                    f_score=tentative_g_score+self.real_heuristic(next_state)
                    parents[next_state]=current_state
                    if next_state not in self.real_open_set.queue:
                        self.real_open_set.put((f_score,next_state))

        return "No path found"
    def reconstruct_path(self,parents, goal_state):
        path = [goal_state]
        current_state = goal_state

        while current_state in parents:
            current_state = parents[current_state]
            path.append(current_state)

        return path[::-1]  # Reverse the path to get it from start to goal
    
    def reconstruct_real_path(self,parents, goal_state):
        path = [goal_state]
        current_state = goal_state

        while current_state in parents:
            current_state = parents[current_state]
            path.append(current_state)

        return path[::-1]
    def policy_initialization(self,house_map): 
        for i in range(100):
            for j in range(100):
                delta_x,delta_y=self.goal_state.row-i,self.goal_state.col-j
                for k in range(4):
                    for l in range(4):
                        if is_collision(house_map=house_map,robot_state=RobotState(row=i,col=j,direction=k,speed=l)):
                            self.pi_0[i,j,k,l]=self.action_encoder(-1,0) #一定先要减速
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
        gamma=1.5
        c=0.8
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
                Q_s_a.append(intermediate(self.Q[robot_state_tupple[0],robot_state_tupple[1],robot_state_tupple[2],robot_state_tupple[3],iter(action_available).__next__()],N_s,n_s_a,c=c))
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
        alpha=0.37

        if d==0:
            return 0
        """
        #根据rollout policy采样一个action,非常值得修正
        preferred_action=self.pi_0[int(robot_state.row),int(robot_state.col),int(robot_state.direction),int(robot_state.speed)]
        #寻找目前可以用的action
        """
        action_available=self.action_available(robot_state)
        probabilities=[]
        for item in action_available:
            probabilities.append(1.0/len(action_available))
        #归一化
        probabilities=np.array(probabilities)
        probabilities/=np.sum(probabilities)
        a=np.random.choice(action_available,p=probabilities) #引入随机性,为什么概率有时候不是1？？？,这只可能是选中的prefered action有问题

        next_state,error,collision=self.transition(house_map,robot_state,a)
        return self.reward_function(house_map,robot_state,a,next_state,error,collision)+self.rollout(house_map,next_state,d-1)
        
        
    def reward_function(self,house_map,robot_state,action,next_robot_state,error,collision): #error是指他有没有不按照指令前进,collision是指过程中是否发生了碰撞
        decoded_action=self.action_decoder(action)
        #到达真终点则有超级大reward
        if next_robot_state.row==self.goal_state.row and next_robot_state.col==self.goal_state.col and next_robot_state.speed==0:
            return 1000.0
        #真终点附近保持速度为1有奖励
        Euclidean_distance_to_final=((next_robot_state.row-self.goal_state.row)**2+(next_robot_state.col-self.goal_state.col)**2)**0.5
        if Euclidean_distance_to_final<5:
            if next_robot_state.speed==1:
                return 400.0
            if robot_state.speed==0 and decoded_action[0]==1:
                return 400.0
            if robot_state.speed>1 and decoded_action[0]==-1:
                return 400.0
        
        
        reward=0.0
        
        current_goal=self.current_goal_in_path[-1]
        current_goal=self.state_decoder(current_goal)
        if current_goal[0]==next_robot_state.row and current_goal[1]==next_robot_state.col:
            return 1000.0
        #先改速度，和方向，再动 
        #计算一下Manhattan距离
        next_delta_x,next_delta_y=current_goal[0]-next_robot_state.row,current_goal[1]-next_robot_state.col
        next_Manhattan_distance=abs(next_delta_x)+abs(next_delta_y)
        next_Euclidean_distance=((next_delta_x)**2+(next_delta_y)**2)**0.5
        current_delta_x,current_delta_y=current_goal[0]-robot_state.row,current_goal[1]-robot_state.col
        current_Manhattan_distance=abs(current_delta_x)+abs(current_delta_y)
        current_Euclidean_distance=((current_delta_x)**2+(current_delta_y)**2)**0.5
        #减小manhattan距离将会有reward
        if next_Euclidean_distance<=current_Euclidean_distance+1:
            reward+=80.0*(current_Euclidean_distance-next_Euclidean_distance+1) #这保障了不会出现除0的情况,分母也不是0，30为超参数
            #在减小的基础上，如果加速，应该reward更大一点
            #reward+=10.0*abs(next_robot_state.speed-robot_state.speed)
            #如果距离已经很小了，在减小的基础上，那么就应该更大的reward
            if next_Euclidean_distance<6:
                reward+=220.0
            #在离目标进的时候应该控制速度为1
            if next_Euclidean_distance<4:
                if robot_state.speed>1: #and decoded_action[0]==-1:
                    reward+=100.0
                elif robot_state.speed==0: #and decoded_action[0]==1:
                    reward+=100.0
                elif robot_state.speed==1: #and decoded_action[0]==0:
                    reward+=100.0
                else:
                    reward-=100.0
        #改变方向的reward或者速度的reward(走了一步所以给点reward)，我考虑给一个随机的reward，以克服我不知道reward的问题
        #reward+=30.0*random.random()
        if next_Manhattan_distance<=4:
            #保持坐标与终点一样会有奖励
            if robot_state.row==self.goal_state.row:
                reward+=200.0
            if robot_state.col==self.goal_state.col:
                reward+=200.0
        #增大距离
        if next_Manhattan_distance>current_Manhattan_distance:
            reward=-100.0
        #暂时到这里,可以接着完善
        
        #截断reward
    
        #如果他并不是按照指令前进的，那么reward应该打折扣，以概率为折扣依据
    

        #如果他碰撞了，那么reward更应小
        if collision==1:
            reward=-300.0 #最后一个是150到250
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
        self.steps_counter+=1
        search_depth=4
        search_times=90
        self.robot_state=robot_state
        self.start_time=datetime.datetime.now()
        """
        if(self.steps_counter==25):
            #清空Q和N和T
            self.Q=np.zeros((100,100,4,4,9))
            self.N=np.zeros((100,100,4,4,9))
            self.T=[]
            #当前state加入T
            self.T.append((robot_state.row,robot_state.col,robot_state.direction,robot_state.speed))
            self.steps_counter=0
        """
        
        if(self.initialization==1):
            self.initialization=0
            #初始化
            self.start_state=robot_state
            #遍历地图以找出终点
            for i in range(100):
                for j in range(100):
                    if house_map[i][j]==3:
                        self.goal_state=RobotState(row=i-1,col=j,direction=0,speed=0)
                        break
            """
            self.T.append((self.start_state.row,self.start_state.col,self.start_state.direction,self.start_state.speed)) #根节点不进行rollout??
            error_prob=0.1
            for i in range(4):
                for j in range(4):
                    for k in range(9):
                        #应该去的状态
                        decoded_action=self.action_decoder(k)
                        v=i+decoded_action[0]
                        v=max(min(v,3),0)
                        d=(j+decoded_action[1])%4
                        for l in range(4):
                            for m in range(4):
                                if l==v and m==d:
                                    self.Transition[i,j,k,l,m]=1-error_prob #应该去的状态,概率大
                                else:
                                    self.Transition[i,j,k,l,m]=error_prob/15 #其他状态,概率小
            """
            #初始化A*算法
            self.house_map=house_map
            #计算用时
            self.start_time=datetime.datetime.now()
            self.path=self.astar(self.state_encoder(self.start_state),self.state_encoder(self.goal_state))
            end=datetime.datetime.now()
            print("A*算法用时：",end-self.start_time) 
            #self.path_mender(house_map,robot_state)
            #定义一个进度，沿着这个路走的进度
            self.current_position_in_path=0
            
            self.initialization=0

        self.Euclidean_distance_to_final=((robot_state.row-self.goal_state.row)**2+(robot_state.col-self.goal_state.col)**2)**0.5
        
        
        if self.steps_counter==20 and self.Euclidean_distance_to_final>10:
            #重新计算一遍路径
            self.path=self.astar(self.state_encoder(self.start_state),self.state_encoder(self.goal_state))
            self.current_position_in_path=0
            self.steps_counter=0

        #先更新一下当前的目标
        if self.path=="No path found":
            #先执行简单policy
            wrong_direction=1
            current_goal_position=(self.goal_state.row,self.goal_state.col)
            delta_row=current_goal_position[0]-robot_state.row
            delta_col=current_goal_position[1]-robot_state.col
            direction_one=0
            direction_two=0
            if abs(delta_row)>=abs(delta_col):
                #先走row
                if robot_state.direction==1 and delta_row<0:
                    wrong_direction=0 
                    direction_one=1
                if robot_state.direction==3 and delta_row>0:
                    wrong_direction=0
                    direction_one=3
                #先走col
                if robot_state.direction==0 and delta_col<0:
                    wrong_direction=0
                    direction_two=0
                if robot_state.direction==2 and delta_col>0:
                    wrong_direction=0
                    direction_two=2
            else:
                if robot_state.direction==1 and delta_row<0:
                    wrong_direction=0 
                    direction_two=1
                if robot_state.direction==3 and delta_row>0:
                    wrong_direction=0
                    direction_two=3
                #先走col
                if robot_state.direction==0 and delta_col<0:
                    wrong_direction=0
                    direction_one=0
                if robot_state.direction==2 and delta_col>0:
                    wrong_direction=0
                    direction_one=2
            
            if wrong_direction:
                if robot_state.speed==0: #先转向到one,如果one前面有墙，就转向two
                    action=(0,1)
                else: #先减速
                    action=(-1,0)
            else: #加速
                #但是如果前面是墙，需要转到另一个方向
                if robot_state.direction==0:
                    next_position=[robot_state.row,robot_state.col-robot_state.speed]
                elif robot_state.direction==1:
                    next_position=[robot_state.row-robot_state.speed,robot_state.col]
                elif robot_state.direction==2:
                    next_position=[robot_state.row,robot_state.col+robot_state.speed]
                elif robot_state.direction==3:
                    next_position=[robot_state.row+robot_state.speed,robot_state.col]
                if robot_state.direction==direction_one:
                    #检测一下撞墙的情况
                    if is_collision(house_map,RobotState(row=next_position[0],col=next_position[1],direction=robot_state.direction,speed=robot_state.speed)): #排除掉direction_one
                        if robot_state.speed==0:
                            action=(0,1)
                        else:
                            action=(-1,0)
                    else:
                        action=(1,0)
                else:
                    action=(1,0)
            
            self.initialization=1
            return Action(acc=action[0],rot=action[1])
            

        self.get_goal_in_path(house_map,robot_state)
        self.current_goal_in_path
        self.current_position_in_path
        self.path
        #计算现在距离最终目标的距离
        
       
        #从当前状态开始计算
        current_goal_position=self.current_goal_in_path[-1]
        current_goal_position=self.state_decoder(current_goal_position)
        current_goal_state_set=[]
        for i in range(4):
            for j in range(4):
                current_goal_state=RobotState(row=current_goal_position[0],col=current_goal_position[1],direction=i,speed=j)

                current_goal_state_set.append(self.real_state_encoder(current_goal_state))
        #用A*算法计算出一条路径
        
        #分两种情况，第一个是距离很远的时候，采用A*算法
        if(self.Euclidean_distance_to_final>8):
            #记录时间
            self.real_path=self.real_astar(self.real_state_encoder(robot_state),current_goal_state_set)
        else:
            #一共4个最后终点态
            row,col=self.goal_state.row,self.goal_state.col
            final=[row*1600+col*16+i*4+0 for i in range(4)]
            self.real_path=self.real_astar(self.real_state_encoder(robot_state),final)
        
        if self.real_path=="No path found" or len(self.real_path)==1: #no path 代表超时间了
            #采用mcts算法
            now=datetime.datetime.now()
            print("现在时间：",now-self.start_time)
            print("超时")
            """
            for i in range(search_times):
                self.simulate(house_map,robot_state,search_depth)
            action_available=self.action_available(robot_state)
            Q_s_a=[self.Q[robot_state.row,robot_state.col,robot_state.direction,robot_state.speed,action] for action in action_available]
            a=np.argmax(Q_s_a)
            action=self.action_decoder(action_available[a])
            """
            #采用简单policy
            #判断目前位置与目标的位置关系和方向区别,走最长的那条路
            wrong_direction=1
            current_goal_position=self.current_goal_in_path[-1]
            current_goal_position=self.state_decoder(current_goal_position)
            delta_row=current_goal_position[0]-robot_state.row
            delta_col=current_goal_position[1]-robot_state.col
            direction_one=0
            direction_two=0
            if abs(delta_row)>=abs(delta_col):
                #先走row
                if robot_state.direction==1 and delta_row<0:
                    wrong_direction=0 
                    direction_one=1
                if robot_state.direction==3 and delta_row>0:
                    wrong_direction=0
                    direction_one=3
                #先走col
                if robot_state.direction==0 and delta_col<0:
                    wrong_direction=0
                    direction_two=0
                if robot_state.direction==2 and delta_col>0:
                    wrong_direction=0
                    direction_two=2
            else:
                if robot_state.direction==1 and delta_row<0:
                    wrong_direction=0 
                    direction_two=1
                if robot_state.direction==3 and delta_row>0:
                    wrong_direction=0
                    direction_two=3
                #先走col
                if robot_state.direction==0 and delta_col<0:
                    wrong_direction=0
                    direction_one=0
                if robot_state.direction==2 and delta_col>0:
                    wrong_direction=0
                    direction_one=2
            
            if wrong_direction:
                if robot_state.speed==0: #先转向到one,如果one前面有墙，就转向two
                    action=(0,1)
                else: #先减速
                    action=(-1,0)
            else: #加速
                #但是如果前面是墙，需要转到另一个方向
                if robot_state.direction==0:
                    next_position=[robot_state.row,robot_state.col-robot_state.speed]
                elif robot_state.direction==1:
                    next_position=[robot_state.row-robot_state.speed,robot_state.col]
                elif robot_state.direction==2:
                    next_position=[robot_state.row,robot_state.col+robot_state.speed]
                elif robot_state.direction==3:
                    next_position=[robot_state.row+robot_state.speed,robot_state.col]
                if robot_state.direction==direction_one:
                    #检测一下撞墙的情况
                    if is_collision(house_map,RobotState(row=next_position[0],col=next_position[1],direction=robot_state.direction,speed=robot_state.speed)): #排除掉direction_one
                        if robot_state.speed==0:
                            action=(0,1)
                        else:
                            action=(-1,0)
                    else:
                        action=(1,0)
                else:
                    action=(1,0)
            return Action(acc=action[0],rot=action[1])
        

        if self.action_list.__len__()==0 or self.steps_counter==1: #为了不要前后A星的计划冲突太多,但是由于有概率出错所以还得判断当前action能不能用上,以免出现illegal的action
            self.action_list=[]
            self.steps_counter=0
            for i in range(1,len(self.real_path)):
                next_state=self.real_state_decoder(self.real_path[i])
                last_state=self.real_state_decoder(self.real_path[i-1])
                #判断这是哪个动作
                if next_state[3]!=last_state[3]:
                    acc=next_state[3]-last_state[3]
                    rot=0
                else:
                    if next_state[2]==last_state[2]:
                        acc=0
                        rot=0
                    else:
                        acc=0
                        if next_state[2]==3 and last_state[2]==0:
                            rot=-1
                        elif next_state[2]==0 and last_state[2]==3:
                            rot=1
                        else:
                            rot=next_state[2]-last_state[2]
                action=Action(acc=acc,rot=rot)
                self.action_list.append(action)
        action=self.action_list.pop(0)
        #判断这个action是否合法,不合法必须重新制作action_list,并且更正
        legal=1
        if robot_state.speed==0:
            legal=1
        elif action.rot!=0:
            legal=0
        
        if legal==0:
            self.action_list=[]
            self.steps_counter=0
            for i in range(1,len(self.real_path)):
                next_state=self.real_state_decoder(self.real_path[i])
                last_state=self.real_state_decoder(self.real_path[i-1])
                #判断这是哪个动作
                if next_state[3]!=last_state[3]:
                    acc=next_state[3]-last_state[3]
                    rot=0
                else:
                    if next_state[2]==last_state[2]:
                        acc=0
                        rot=0
                    else:
                        acc=0
                        if next_state[2]==3 and last_state[2]==0:
                            rot=-1
                        elif next_state[2]==0 and last_state[2]==3:
                            rot=1
                        else:
                            rot=next_state[2]-last_state[2]
                action=Action(acc=acc,rot=rot)
                self.action_list.append(action)
            action=self.action_list.pop(0)

        return Action(acc=action.acc,rot=action.rot)  #return the action for execution
    
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
        encoder=lambda v,d:v*4+d
        decoder=lambda n:(n//4,n%4)
        probabilities=[self.Transition[int(robot_state.speed),int(robot_state.direction),int(action),decoder(i)[0],decoder(i)[1]] for i in range(16)]
        next_encoded_v_d=np.random.choice(range(16),p=probabilities) #概率生成随机状态
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
                    #速度变为0
                    intermediate_state.speed=0
                    break
        
        #防止越界
        if intermediate_state.row<0:
            intermediate_state.row=0
        if intermediate_state.row>99:
            intermediate_state.row=99
        if intermediate_state.col<0:
            intermediate_state.col=0
        if intermediate_state.col>99:
            intermediate_state.col=99
    
        next_state=intermediate_state.copy()
        return (next_state,error,collision)
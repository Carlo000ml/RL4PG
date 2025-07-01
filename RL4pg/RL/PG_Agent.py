
from grid2op.Agent import BaseAgent
class PG_Agent(BaseAgent):

    def __init__(self,environment, ma_manager,gp_manager, line_agents):

        super().__init__(action_space=environment.action_space)
        self.ma_manager=ma_manager
        self.agents=line_agents
        self.gp_manager=gp_manager

    def act(self,observation, reward=None, done=None):
        obs=observation[0]
        graph=observation[1]
        safe_state = self.ma_manager.safe(obs)

        if safe_state:
            return (-1,self.ma_manager.do_nothing())
        else:
            if self.ma_manager.check_disconnections(obs):
                return (-1,self.ma_manager.reconnect_line(obs))
                
            else:
                line_id = self.ma_manager.select_candidate_agent(obs)
                action = self.agents[line_id].act(graph, current_obs=obs)
                return (line_id,action)
            
    def exploit(self,observation, reward=None, done=None):
        obs=observation[0]
        graph=observation[1]
        safe_state = self.ma_manager.safe(obs)

        if safe_state:
            return (-1,self.ma_manager.do_nothing())
        else:
            if self.ma_manager.check_disconnections(obs):
                return (-1,self.ma_manager.reconnect_line(obs))
                
            else:
                line_id = self.ma_manager.select_candidate_agent(obs)
                action = self.agents[line_id].exploit(graph, current_obs=obs)
                return (line_id,action)
            
    def store_experience(self, agent_id, graph, action, reward, next_graph, done):
        self.agents[agent_id].store_experience(graph=graph, action=action, reward=reward, next_graph=next_graph, done=done)

    def learn(self):
        for a in self.agents:
            a.learn()

    def sync(self):
        self.gp_manager.sync_target()
        for a in self.agents:
            a.estimator_manager.sync_target()


            

        

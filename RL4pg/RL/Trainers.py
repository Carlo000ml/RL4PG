from torch_geometric.data import Data
import numpy as np

def exploit_episode(Agents, MA_Controller, Env, options={} ):
    obs,graph = Env.reset(options=options)

    done = False
    ep_len = 0
    cum_rew = 0
    do_action=0

    do_n_action=Env.grid2op_env.action_space({}).as_dict()

    while not done:
        safe_state = MA_Controller.safe(obs)

        if safe_state:
            (next_obs, next_graph), reward, done, info = Env.step(MA_Controller.do_nothing())
        else:
            if MA_Controller.check_disconnections(obs):
                action = MA_Controller.reconnect_line(obs)
                (next_obs, next_graph), reward, done, info = Env.step(action)
            else:
                # select the agent
                sub_id = MA_Controller.exploit_agent(obs)
                # action
                action = Agents[sub_id].exploit(graph, current_obs=obs)
                if not action.as_dict() == do_n_action:
                    do_action+=1
                (next_obs, next_graph), reward, done, info = Env.step(action)
                reward=np.average(   (   np.maximum(1-next_obs.rho,  0)   )**2   ) if not done else -1

        cum_rew += reward
        ep_len += 1
        obs=next_obs
        graph=next_graph

    return {"cum reward": cum_rew , "survive time": ep_len, "Do action" :do_action }




def train_episode(Agents, MA_Controller, Env, reward_decomposer=True, options={}):
    obs,graph = Env.reset(options=options)
    n_sub=obs.n_sub

    done = False
    ep_len = 0
    cum_rew = 0
    
    agents_counter=[]


    while not done:
        safe_state = MA_Controller.safe(obs)

        if safe_state:
            (next_obs, next_graph), reward, done, info = Env.step(MA_Controller.do_nothing())
        else:
            if MA_Controller.check_disconnections(obs):
                action = MA_Controller.reconnect_line(obs)
                (next_obs, next_graph), reward, done, info = Env.step(action)
            else:
                sub_id,_ = MA_Controller.play_agent(obs)
                action, exploration = Agents[sub_id].act(graph, current_obs=obs)
                (next_obs, next_graph), reward, done, info = Env.step(action)
                line_rew=np.average(   (   np.maximum(1-next_obs.rho,  0)   )**2   ) if not done else -1  # -1 only if done
                Agents[sub_id].store_experience((graph, Agents[sub_id].grid2op_to_torch(action), line_rew, next_graph, done))

                if not exploration:
                    MA_Controller.store_experience(experience=(MA_Controller.obs_to_torch(obs) , sub_id , line_rew, MA_Controller.obs_to_torch(next_obs), done))
                agents_counter.append(sub_id)

        cum_rew += np.average(   (   np.maximum(1-next_obs.rho,  0)   )**2   ) if not done else -1
        ep_len += 1
        obs=next_obs
        graph=next_graph

    

    return {"cum reward": cum_rew , "survive time": ep_len , "agents counter": agents_counter}




def do_nothing_episode(Env, options={}):

        _=Env.reset(options=options)
        done=False
        ep_len=0
        cum_rew=0
    
        
    
        while not done:

            
            (next_obs, next_graph), reward, done, info = Env.step( Env.grid2op_env.action_space({}))
                    
            cum_rew+=np.average(   (   np.maximum(1-next_obs.rho,  0)   )**2   ) if not done else -1
            ep_len+=1

        return {"cum reward": cum_rew , "survive time": ep_len} 



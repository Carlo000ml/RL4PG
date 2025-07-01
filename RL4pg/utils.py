import torch
import numpy as np
from torch_geometric.data import Data
import networkx as nx
from torch_geometric.utils import to_networkx
from collections import defaultdict
import random


def obs_to_torch(obs, device):
    return torch.tensor(np.concatenate([obs.rho, obs.a_or, obs.a_ex, obs.topo_vect],axis=0), device=device, dtype=torch.float32)


def el_to_vect(obs, el, element_type, device='cpu', done=False): 
    """
    Converts the features of a specified power grid element of a specific observation into a fixed-length feature vector.

    This utility function extracts features of a specific element type (e.g., load, generator, line, bus)
    from a grid2op observation and returns a 41-dimensional tensor representation of the element's state.
    Note: storages are not considered for the moment.

    Args:
        obs (grid2op.Observation.Observation): 
            The current grid2op observation containing the power grid's state information.
        el (int): 
            The index of the specific element to extract features for. The index must be relative to its type
            that is: index 0 for element_type="load" returns the vector for the load 0.
        element_type (str): 
            The type of the element. Must be one of ["load", "ren gen", "non ren gen", "line", "bus"].
        device (torch.device): 
            The device (e.g., "cpu" or "cuda") on which to create the output tensor. Default is 'cpu'.
        done (bool, optional): 
            Indicates whether the episode is finished. If True, the feature vector for buses will be zeros. 
            This avoid problems: when the episode is finished the conversion obs.as_networkx() does not include any node.
            Default is False.

    Returns:
        torch.Tensor: 
            A 41-dimensional tensor representing the features of the specified element. The feature 
            vector is pre-filled with zeros, and only the section corresponding to the element type 
            is populated with its specific features.

    Raises:
        AssertionError: 
            If `element_type` is not one of the allowed values: ["load", "ren gen", "non ren gen", "line", "bus"].

    Feature Encoding:
        - The output vector is partitioned into fixed segments for each element type:
            * Load: [0:3]
            * Renewable Generator: [3:12]
            * Non-renewable Generator: [12:28]
            * Line: [28:38]
            * Bus: [38:41]
        - Only the segment corresponding to the specified element type is populated, with the rest remaining zeros.


    Example Usage:
        >> feature_vec = el_to_vect(obs, el=13, element_type="bus", device=torch.device("cpu"))
        >> print(feature_vec.shape)
        torch.Size([41])
    """
    assert element_type in ["load" , "ren gen" , "non ren gen" , "line", "bus"]
    assert len(obs.storage_to_subid)==0, "This environment uses storages"

    feature_vec=torch.zeros(41, device=device, requires_grad=False)

    if element_type=="load": 
        offset=0 
        space=3
    if element_type=="ren gen": 
        offset=3
        space=9
    if element_type=="non ren gen": 
        offset=12
        space=16
    if element_type=="line":
        offset=28
        space=10
    if element_type=="bus":
        offset=38
        space=3


    if element_type=="load": feat=np.hstack([obs.load_p[el],obs.load_q[el],obs.load_v[el]])
    if element_type=="ren gen": feat = np.hstack([obs.gen_p[el], obs.gen_q[el], obs.gen_v[el], obs.gen_pmin[el], obs.gen_pmax[el], obs.gen_p_before_curtail[el], obs.curtailment_limit[el], obs.curtailment_limit_effective[el], obs.curtailment_mw[el]])
    if element_type=="non ren gen": feat = np.hstack([obs.gen_p[el], obs.gen_q[el], obs.gen_v[el], obs.gen_pmin[el], obs.gen_pmax[el], obs.gen_max_ramp_up[el], obs.gen_max_ramp_down[el], obs.gen_min_uptime[el], obs.gen_min_downtime[el], obs.gen_cost_per_MW[el], obs.gen_startup_cost[el], obs.gen_shutdown_cost[el], obs.target_dispatch[el], obs.actual_dispatch[el], obs.gen_margin_up[el], obs.gen_margin_down[el]])
    if element_type=="line":feat = np.hstack([obs.rho[el], obs.line_status[el], obs.p_or[el], obs.q_or[el], obs.v_or[el], obs.a_or[el], obs.p_ex[el], obs.q_ex[el], obs.v_ex[el], obs.a_ex[el]])
    if element_type=="bus": 
        if done:
            feat=np.zeros(3)
        else:
            Net=obs.as_networkx()
            
            matching_key = next((key for key, value in dict(Net.nodes(data="global_bus_id")).items() if value == el), None)
            if matching_key!=None: feat=np.array([Net.nodes(data="p")[matching_key] , Net.nodes(data="q")[matching_key] , Net.nodes(data="v")[matching_key]])
            else: feat=np.zeros(3)
    feat=torch.tensor(feat, device=device, requires_grad=False)


    feature_vec[offset:offset+space]=feat

    return feature_vec



def el_2_vec_bus_shift(obs, el, element_type,bus,  device='cpu', done=False ):
    el_vec=el_to_vect(obs=obs, el=el, element_type=element_type,  device=device, done=done)
    feature_vec=torch.zeros(41* 3, device=device, requires_grad=False)
    mapping={1:0 , 2:1 , -1:2}
    start_idx=41* mapping[bus]
    end_index=start_idx+41
    feature_vec[start_idx:end_index]=el_vec
    return feature_vec








def substation_vectors(obs, device='cpu' , done=False):
    n_sub=obs.n_sub
    substation_embeddings=torch.zeros((n_sub, 41*3),  device=device)


    for id in range(n_sub):   # bus 1 0-14
        substation_embeddings[id,:]+=el_2_vec_bus_shift(obs,id,"bus", bus=1,device=device, done=done)
    
    for id in range(n_sub):   # bus 2  14-28
        substation_embeddings[id,:]+=el_2_vec_bus_shift(obs, n_sub+id,"bus",bus=2,device=device, done=done)

    for id in range(obs.n_load):   
        bus=obs.load_bus[id]
        sub=obs.load_to_subid[id]
        substation_embeddings[sub,:]+=el_2_vec_bus_shift(obs,id,"load" , bus=bus,device=device, done=done)


    for id in range(obs.n_gen):
        bus=obs.gen_bus[id]
        sub=obs.gen_to_subid[id]
        if obs.gen_renewable[id]: 
            substation_embeddings[sub,:]+=el_2_vec_bus_shift(obs,id,"ren gen",bus=bus, device=device, done=done)
        else: 
            substation_embeddings[sub,:]+=el_2_vec_bus_shift(obs,id,"non ren gen",bus=bus , device=device,  done=done)

    for id in range(obs.n_line):  # origin of line
        bus_or=obs.line_or_bus[id]
        bus_ex=obs.line_ex_bus[id]
        sub_or=obs.line_or_to_subid[id]
        sub_ex=obs.line_ex_to_subid[id]
        substation_embeddings[sub_or,:]+=el_2_vec_bus_shift(obs,id,"line",bus=bus_or , device=device,  done=done)
        substation_embeddings[sub_ex,:]+=el_2_vec_bus_shift(obs,id,"line",bus=bus_ex , device=device,  done=done)

    return substation_embeddings





def line_vectors(obs, device='cpu' , done=False):
    n_line=obs.n_line
    sv=substation_vectors(obs,device=device, done=done)
    line_vectors=torch.zeros((n_line, 41*3* 2), device=device)

    for line in range(n_line):
        sub0=obs.line_or_to_subid[line]
        sub1=obs.line_ex_to_subid[line]
        line_vectors[line]=torch.cat([sv[sub0,:] , sv[sub1,: ]] , dim=0)

    return line_vectors


def sub_connectivity(obs, as_dict=False):
    """
    
    Possible problem if multiple lines between two substations
    
    """


    # a tensor of substation extremities for each line
    sub_edge_index=torch.tensor(np.vstack([obs.line_or_to_subid, obs.line_ex_to_subid]),dtype=torch.long)

    if as_dict:  # to have (substation extremities) -> line_id  // here we put also the reversed substation extremities mapped to the same line_id
        d={}
        for i in range(obs.n_line):
            if (sub_edge_index[0][i].item(),sub_edge_index[1][i].item()) not in list(d.keys()):
                d[(sub_edge_index[0][i].item(),sub_edge_index[1][i].item())]=i 
                d[(sub_edge_index[1][i].item(),sub_edge_index[0][i].item())]=i 
            else:
                if type(d[(sub_edge_index[0][i].item(),sub_edge_index[1][i].item())])==int:
                     d[(sub_edge_index[0][i].item(),sub_edge_index[1][i].item())]=[d[(sub_edge_index[0][i].item(),sub_edge_index[1][i].item())] , i]
                else:
                     d[(sub_edge_index[0][i].item(),sub_edge_index[1][i].item())].append(i)



        return d

    return sub_edge_index


def line_connectivity(obs, multigraph=False):
    """
    set multigraph=True if multiple lines between two substations

    """
    sub_edge_index=sub_connectivity(obs)
    mapping=sub_connectivity(obs, as_dict=True)

    # pytorch graph
    sub_graph=Data(edge_index=sub_edge_index, num_nodes=obs.n_sub)

    # conversion to networksx graph  -> undirected
    nx_graph=to_networkx(sub_graph, to_undirected=True, to_multi=multigraph)
    if multigraph:
        mapping={}
        for edg in nx_graph.edges:
            sub0=edg[0]
            sub1=edg[1]
            lineids=np.where(np.all(  (np.transpose(sub_connectivity(obs).numpy())==np.array([sub0,sub1])) |   (np.transpose(sub_connectivity(obs).numpy())==np.array([sub1,sub0]))  ,axis=1))[0] 
            for k in range(len(lineids)):

                mapping[(sub0,sub1,k)]=lineids[k]
                mapping[(sub1,sub0,k)]=lineids[k]

    # build the line graph  -> built in function
    line_g=nx.line_graph(nx_graph)


    edg=[]
    for e in line_g.edges:
        edg.append((mapping[e[0]] , mapping[e[1]]))
        edg.append((mapping[e[1]] , mapping[e[0]]))
    return torch.tensor(edg).T



def build_torch_line_graph(obs, multigraph=False, device='cpu'):
    x=line_vectors(obs, device=device)
    edge_index=line_connectivity(obs,multigraph=multigraph).to(device)
    return Data(x=x, edge_index=edge_index)




def build_line_extremities_sub_indexes(obs):
    
    return torch.tensor(np.vstack([obs.line_or_to_subid, obs.line_ex_to_subid]) , dtype=torch.long ).t().contiguous()



def build_line_action_space(env,line_id, action_type="set"):
    line_extremities=build_line_extremities_sub_indexes(env)[line_id]

    if action_type=="set":
        action_space=env.action_space.get_all_unitary_topologies_set(env.action_space, sub_id=line_extremities[0].item(),add_alone_line=False) +env.action_space.get_all_unitary_topologies_set(env.action_space, sub_id=line_extremities[1].item(),add_alone_line=False)+[env.action_space({})]
    elif action_type=="change":
        action_space=env.action_space.get_all_unitary_topologies_change(env.action_space, sub_id=line_extremities[0].item(),add_alone_line=False) +env.action_space.get_all_unitary_topologies_change(env.action_space, sub_id=line_extremities[1].item(),add_alone_line=False)+[env.action_space({})]

    return action_space
    

def fake_action(current_obs,act):
    """
    return true if the action is not performing anything
    """
    affected_elements=np.nonzero(act._set_topo_vect)

    return np.all(current_obs.topo_vect[affected_elements]==act._set_topo_vect[affected_elements])

def bad_action(obs, act):
    """
    return true if the action will increase the maximum rho of the power grid
    """
    current_simulation = obs.get_simulator()

    current_max_rho=np.max(obs.rho)
    new_max_rho=np.max(current_simulation.predict(act).current_obs.rho)
    return new_max_rho> current_max_rho

def action_masking(action_space, obs):
    """
    return the indexes of fake actions
    """
    return torch.tensor(np.nonzero(np.array([fake_action(obs, a) for a in action_space]))[0][:-1])

def bad_action_masking(action_space, obs):
    """
    return indexes of bad actions
    """
    return torch.tensor(np.nonzero(np.array([bad_action(obs, a) for a in action_space]))[0])


def merge_dicts(*dicts):
    """Merges multiple dictionaries where values are lists."""
    merged = defaultdict(list)
    for d in dicts:
        for key, value in d.items():
            merged[key].extend(value)  # Efficient list concatenation
    return dict(merged)




class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity  # Maximum number of leaf nodes (experiences)
        self.tree = np.zeros(2 * capacity - 1)  # Complete binary tree
        self.data = np.empty(capacity, dtype=object)  # To store experiences
        self.write = 0  # Pointer for the next data index
        self.size = 0

    def add(self, priority, exp):
        idx = self.write + self.capacity - 1  # Compute the tree index for the new data
        self.data[self.write] = exp          # Store the experience
        self.update(idx, priority)            # Update the tree with the new priority

        # Move the write pointer cyclically
        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        # Propagate the change up to the root
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get_leaf(self, value):
        """
        Traverse the tree to find the leaf index, its priority, and the stored data,
        given a random value in [0, total_priority].
        """
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            # If we reach a leaf, stop
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if value <= self.tree[left_child_idx]:
                    parent_idx = left_child_idx
                else:
                    value -= self.tree[left_child_idx]
                    parent_idx = right_child_idx
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]
    
    def __repr__(self):
        def build_tree(idx, prefix="", is_tail=True):
            left_idx = 2 * idx + 1
            right_idx = 2 * idx + 2

            # If we reached a leaf node, show the (priority, object)
            if left_idx >= len(self.tree):
                data_idx = idx - self.capacity + 1
                return prefix + ("└── " if is_tail else "├── ") + f"({self.tree[idx]}, {self.data[data_idx]})\n"
            else:
                # For internal nodes, show the sum stored at the node.
                result = prefix + ("└── " if is_tail else "├── ") + f"{self.tree[idx]}\n"
                # Prepare the prefix for the children.
                new_prefix = prefix + ("    " if is_tail else "│   ")
                result += build_tree(left_idx, new_prefix, False)
                result += build_tree(right_idx, new_prefix, True)
                return result

        return build_tree(0, "", True)


    @property
    def total_priority(self):
        return self.tree[0]  # The root holds the total priority






def collect_n_step_expert_experiences(episode,n,action_converters,build_graph, experiences={i: [] for i in range(20)}, device="cpu"):
     # iterate over all the time steps
    for time_step in range(len(episode.actions)-n):
        # action performed a_t
        action=episode.actions[time_step]
        # if it is not a do nothing action AND it is not a line reconnection/disconnection
        if not action.as_dict() == {} and np.all(action.set_line_status==0):
            # obs t
            obs=episode.observations[time_step]
            graph=build_graph(obs, device=device)

            # line id in danger
            line_id=np.argmax(obs.rho)

            # sub id on which the action has been performed
            #print(time_step)
            sub_id=int(action.as_dict()["set_bus_vect"]["modif_subs_id"][0])

            # check if the action belongs to the action space of the line agent -> if the sub_id is a sub extremity of the line in danger + check if the action is in its action space
            if sub_id in action_converters[line_id].line_extremities:
                done_t=0
                done_t_n=0
                rewards=[]
            
                # 1 step
                if action.as_dict() in action_converters[line_id].action_space:
                    agent_action=action_converters[line_id].grid2op_to_torch(action)
                    next_obs=episode.observations[time_step+1]
                    next_graph=build_graph(next_obs, device=device)
                    next_n_graph=next_graph # to allow setting n=1 (single step loss)
                    agent_reward=np.average(   (   np.maximum(1-next_obs.rho,  0)   )**2   )
                    if agent_reward==1:
                            done_t=1
                    rewards.append(agent_reward)
                    
                    # following steps
                    for i in range(2,n):
                        next_obs=episode.observations[time_step+i]
                        next_n_graph=build_graph(next_obs, device=device)
                        agent_next_reward=np.average(   (   np.maximum(1-next_obs.rho,  0)   )**2   ) # starting from reward t+1  until reward t+n-1
                        rewards.append(agent_next_reward)
                        if agent_next_reward==1:
                            done_t_n=1
                        
                    experiences[line_id].append((graph, agent_action.item(), torch.tensor(rewards), done_t, next_graph, next_n_graph, done_t_n))
    return experiences
    


def collect_n_step_expert_experiences_RL_Manager(episode,n,action_converters,build_graph, experiences={i: [] for i in range(20)}, MA_exp=[], device="cpu"):
     # iterate over all the time steps
    for time_step in range(len(episode.actions)-n):
        # action performed a_t
        action=episode.actions[time_step]
        # if it is not a do nothing action AND it is not a line reconnection/disconnection
        if not action.as_dict() == {} and np.all(action.set_line_status==0):
            # obs t
            obs=episode.observations[time_step]
            graph=build_graph(obs, device=device)
            state=obs_to_torch(obs, device=device)

            # sub id on which the action has been performed
            #print(time_step)
            sub_id=int(action.as_dict()["set_bus_vect"]["modif_subs_id"][0])

            # compute all the lines id connected to that substation
            agents_selected=np.concatenate((np.where(obs.line_or_to_subid==sub_id)[0], np.where(obs.line_ex_to_subid==sub_id)[0]))

            for ag in agents_selected:
                done_t=0
                done_t_n=0
                rewards=[]
                # check if the action belongs to its action_space
                if action.as_dict() in action_converters[ag].action_space:
                    agent_action=action_converters[ag].grid2op_to_torch(action)

                    next_obs=episode.observations[time_step+1]
                    next_graph=build_graph(next_obs, device=device)
                    next_state=obs_to_torch(next_obs, device=device)
                    next_n_graph=next_graph
                    next_n_state=next_state
                    agent_reward=np.average(   (   np.maximum(1-next_obs.rho,  0)   )**2   )
                    if agent_reward==1:
                            done_t=1
                    rewards.append(agent_reward)
                    
                    # following steps
                    for i in range(2,n):
                        next_obs=episode.observations[time_step+i]
                        next_n_graph=build_graph(next_obs, device=device)
                        next_n_state=obs_to_torch(next_obs, device=device)
                        agent_next_reward=np.average(   (   np.maximum(1-next_obs.rho,  0)   )**2   ) # starting from reward t+1  until reward t+n-1
                        rewards.append(agent_next_reward)
                        if agent_next_reward==1:
                            done_t_n=1

                    experiences[ag].append((graph, agent_action.item(), torch.tensor(rewards), done_t, next_graph, next_n_graph, done_t_n))
                    MA_exp.append((state,ag,torch.tensor(rewards),next_state,done_t,next_n_state, done_t_n))
    return experiences, MA_exp
import gym
from grid2op.Environment import SingleEnvMultiProcess

class GlobalEnvironment(gym.Wrapper):
    def __init__(self, grid2op_env, build_graph, device='cpu'):

        self.grid2op_env = grid2op_env
        self.build_graph = build_graph  # Function to create a graph from the global observation
        self.device=device

    def reset(self, options={}):

        # Reset the Grid2Op environment
        obs = self.grid2op_env.reset(options=options)

        # Convert the global observation to a graph
        graph = self.build_graph(obs, device=self.device)


        return (obs, graph)

    def step(self, action):

        # Play the action in the Grid2Op environment
        obs, reward, done, info = self.grid2op_env.step(action)

        # Convert the new global observation to a graph
        graph = self.build_graph(obs, device=self.device)

        return (obs, graph) , reward , done, info
    



class GlobalMultiProcessEnvironment(gym.Wrapper):
    def __init__(self, grid2op_env,nb_env, build_graph, device='cpu'):

        self.grid2op_env = SingleEnvMultiProcess(env=grid2op_env, nb_env=nb_env)
        self.build_graph = build_graph  # Function to create a graph from the global observation
        self.device=device
        self.nb_env=nb_env

    def reset(self):

        # Reset the Grid2Op environment
        obss = self.grid2op_env.reset()

        observations=[(obss[i], self.build_graph(obss[i], device=self.device)) for i in range(len(obss))]

        return observations

    def step(self, action):

        # Play the action in the Grid2Op environment
        obss, rewards, dones, _ = self.grid2op_env.step(action)

        observations=[(obss[i], self.build_graph(obss[i], device=self.device)) for i in range(len(obss))]

        return observations , rewards , dones, _
    
    def close(self):
        self.grid2op_env.close()
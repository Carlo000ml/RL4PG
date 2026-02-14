from RL4pg.RL.DeepQL.Agents import DQN_Agent
from RL4pg.utils import *
from RL4pg.RL.Trainers import *
from RL4pg.RL.Environments import GlobalEnvironment
from RL4pg.RL.Managers import MultiAgent_RL_Sub_Manager
from RL4pg.Initialize_Env import initialize_env
from RL4pg.RL.Converters import  Action_Converters_sub
from RL4pg.Graph_Processing.GP_Manager import GP_Manager
import torch
import random
import numpy as np
import json
from tqdm import tqdm
from lightsim2grid import LightSimBackend
from grid2op.Reward import L2RPNReward
from grid2op.Action import PlayableAction
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import grid2op
import argparse
import pickle
from grid2op.Episode import EpisodeData

import pathlib
from dotenv import load_dotenv
load_dotenv()



os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"



def set_seed(seed):
    """
    Set seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.use_deterministic_algorithms(True)

def load_config(config_path):
    """
    Load the experiment configuration from a JSON file.
    """
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def main(settings):
    """
    Main function to execute the experiment based on the configuration.

    Running instructions:
    python -m Project.Main_Line_Agents --config <path_to_config_file>

    Args:
        config (dict): Configuration dictionary loaded from JSON.
    """

    policy_kargs=settings["policy_kargs"] 

    Q_value_estimator_kargs=settings["Q_value_estimator_kargs"]
    
    agents_kargs=settings["agents_kargs"]
    
    MA_manager_kargs=settings["MA_manager_kargs"]
    

    config=settings["config"]

    gp_kargs=settings["gp_kargs"]
    
    replay_buffer_kargs=settings["replay_buffer_kargs"]

    project_root= pathlib.Path(os.getenv("PROJECT_ROOT"))
    experiences_path=project_root / "ExpertExperiences"




        # Set seed for reproducibility
    seed = config["environment"].get("seed", 42)
    set_seed(seed)
    if torch.cuda.is_available() and config["device"]=="cuda":
        device = torch.device("cuda")
        print(">> >> using cuda")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024 ** 3:.2f} GB")
        print("Memory Allocated:", torch.cuda.memory_allocated() / 1024 ** 3, "GB") 
        print("Memory Cached:", torch.cuda.memory_reserved() / 1024 ** 3, "GB")
    else:
        device = torch.device("cpu")
        print(f">> >> using cpu")
    gp_kargs["device"]=device

    

    # Create the environment
    env_name = config["environment"]["name"]
    initialize_env(env_name)


    env_train = grid2op.make(
        f"{env_name}_train", backend=LightSimBackend(), reward_class=L2RPNReward, action_class=PlayableAction
    )
    env_train.seed(seed)

    gp_kargs["grid2openv"]= env_train  

    env_val = grid2op.make(
        f"{env_name}_val", backend=LightSimBackend(), reward_class=L2RPNReward, action_class=PlayableAction
    )
    env_val.seed(seed)
    env_test = grid2op.make(
        f"{env_name}_test", backend=LightSimBackend(), reward_class=L2RPNReward, action_class=PlayableAction
    )
    env_test.seed(seed)

    paths=env_val.chronics_handler.subpaths
    val_chr_ids = [int(os.path.basename(path)) for path in paths]

    paths=env_test.chronics_handler.subpaths
    test_chr_ids = [int(os.path.basename(path)) for path in paths]





    init_obs = env_train.reset()
    

    runs_folder=config["hyperparameters"]["save folder"]+ f'Experiment_{datetime.now().strftime("%Y-%m-%d_%H-%M")}'+"_"+config["hyperparameters"].get("run name", "")


    # Initialize TensorBoard writers
    writer = SummaryWriter(log_dir=runs_folder + "/Episode statistics")
    writer2 = SummaryWriter(log_dir=runs_folder + "/Hyperparameters")

    # Hyperparameters from the config file
    hyperparameters = config["hyperparameters"]

    # Log hyperparameters to TensorBoard
    hyperparam_text = "\n".join([f"{key}: {value}" for key, value in hyperparameters.items()])
    agents_kargs_text = "\n".join([f"{key}: {value}" for key, value in agents_kargs.items()])
    policy_kargs_text = "\n".join([f"{key}: {value}" for key, value in policy_kargs.items()])
    Q_value_estimator_kargs_text = "\n".join([f"{key}: {value}" for key, value in Q_value_estimator_kargs.items()])
    MA_manager_kargs_text = "\n".join([f"{key}: {value}" for key, value in MA_manager_kargs.items()])
    gp_kargs_text = "\n".join([f"{key}: {value}" for key, value in gp_kargs.items()])
    replay_buffer_kargs_text = "\n".join([f"{key}: {value}" for key, value in replay_buffer_kargs.items()])
    writer2.add_text("agents_kargs", agents_kargs_text)
    writer2.add_text("Hyperparameters", hyperparam_text)
    writer2.add_text("policy_kargs", policy_kargs_text)
    writer2.add_text("Q_value_estimator_kargs", Q_value_estimator_kargs_text)
    writer2.add_text("MA_manager_kargs", MA_manager_kargs_text)
    writer2.add_text("gp_kargs", gp_kargs_text)
    writer2.add_text("replay_buffer_kargs", replay_buffer_kargs_text)
    writer2.close()

    # Initialize components
    MA_manager_kargs["environment"]=env_train
    MA_manager_kargs["connect_disconnected_line"]=True
    MA_manager_kargs["runs_name"]=runs_folder


    MultiAgent_Controll = MultiAgent_RL_Sub_Manager(**MA_manager_kargs)

    GlobalEnv = GlobalEnvironment(env_train, build_graph=build_torch_sub_graph, device=device)
    GlobalEnv_val = GlobalEnvironment(env_val, build_graph=build_torch_sub_graph, device=device)
    GlobalEnv_test = GlobalEnvironment(env_test, build_graph=build_torch_sub_graph, device=device)

    gp_manager=GP_Manager(**gp_kargs)


    Sub_Agents = {
            i: DQN_Agent(
                action_space=N_1_secure_action_space(env_train,i),
                agent_id=i,
                runs_name=runs_folder,
                policy_kargs=policy_kargs,
                policy_type=agents_kargs["policy_type"],
                Q_value_estimator_kargs=Q_value_estimator_kargs,
                buffer_type=agents_kargs["buffer_type"],
                gp_manager=gp_manager,
                replay_buffer_kargs=replay_buffer_kargs,
                start_training_capacity=agents_kargs["start_training_capacity"],
                num_training_iters=agents_kargs["num_training_iters"],
                rew_shape=agents_kargs["rew_shape"],
                device=device
                )
             for i in range(init_obs.n_sub)
         }
    

    agents_counter=[i for i in range(env_train.n_sub)]

    if hyperparameters["learn demonstrations"]:

        print(f">>> Starting Demonstrations learning")

        print(f">>> Checking if experiences' dataset already exists")
        experiences_path_file=experiences_path  /"Sub/Experiences.pkl"
        MA_experiences_path_file=experiences_path / "Sub/MA_exp.pkl"


        
        if experiences_path_file.exists() and MA_experiences_path_file.exists():
            print(f"Datasets already exist!")
                # Load the dictionary back from the pickle file
            with open(experiences_path_file, 'rb') as f:
                experiences = pickle.load(f)
            with open(MA_experiences_path_file, 'rb') as f:
                MA_exp = pickle.load(f)
        else:
            print(f"Dataset does not exixts... Creating the dataset...")
            episode_studied = EpisodeData.list_episode(config["environment"]["Expert interaction path"])  # To be configured later for better handling of the path

            a_c={i:Action_Converters_sub(env_train,i) for i in range(env_train.n_sub)}

            experiences={i: [] for i in range(env_train.n_sub)}
            MA_exp=[]
            print(f">>> Collecting demonstrations of {agents_kargs['demonstrations n']} steps")
            for i in tqdm(range(len(episode_studied))):
                this_episode = EpisodeData.from_disk(*episode_studied[i])
                experiences,MA_exp=collect_n_step_expert_experiences_RL_Manager_sub(this_episode,n=agents_kargs["demonstrations n"],action_converters=a_c,build_graph=build_torch_sub_graph, playable_subs=MultiAgent_Controll.playable, experiences=experiences, MA_exp=MA_exp)
            
                    # Create the directory if it doesn't exist

            pathlib.Path(experiences_path_file).parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(MA_experiences_path_file).parent.mkdir(parents=True, exist_ok=True)

            # Pickle (save) the dictionary to a file
            with open(experiences_path_file, 'wb') as f:
                pickle.dump(experiences, f)
            with open(MA_experiences_path_file, 'wb') as f:
                pickle.dump(MA_exp, f)

        print(f">>> Storing demonstrations into agents")
        for i in range(env_train.n_sub):
            for e in experiences[i]:
                Sub_Agents[i].store_demonstration(e)
                agents_counter.append(i)

        print(f">>> Storing demonstrations into manager")
        for e in MA_exp:
            MultiAgent_Controll.store_demonstration(e)



        print(f">>> Learning manager demonstrations for {MultiAgent_Controll.num_training_iters * MA_manager_kargs['n demonstrations training']} iterations")
        for k in range(MA_manager_kargs["n demonstrations training"]):
            MultiAgent_Controll.learn_demonstrations()
            if k % MA_manager_kargs["target_update_freq"]==0:
                MultiAgent_Controll.save_checkpoint()
                MultiAgent_Controll.sync_target()



        print(f">>> Learning agents demonstrations for {130*agents_kargs['n training']} iterations")
        
        for k in range(agents_kargs["n training"]):
            for i in range(env_train.n_sub):
                Sub_Agents[i].learn_demonstrations()
                if  k % agents_kargs["target_update_freq"]==0:
                    if agents_kargs["checkpoints"]:
                        gp_manager.save_checkpoint(path=runs_folder)
                    gp_manager.sync_target()
                    for j in range(env_train.n_sub):
                        if agents_kargs["checkpoints"]:
                            Sub_Agents[j].save_estimator_checkpoint()
                        Sub_Agents[j].estimator_manager.sync_target()
    else: print(f"No demonstrations learning")


    
        

    exploit_step=0
    val_step=0
    do_nothing_step=0
    do_nothing_validation=False
    start_exploit=True
    test=True
    do_nothing_test=False
    test_step=0
    do_nothing_step_test=0

    options={}

    # Run episodes
    for episode in tqdm(range(hyperparameters["number of episodes"])):
        

        if episode%hyperparameters["exploit interval"] < hyperparameters["exploit tests"] and start_exploit:

            exploit_res=exploit_episode(Agents=Sub_Agents,MA_Controller=MultiAgent_Controll, Env=GlobalEnv,options=options)
            writer.add_scalar("Cumulative L2RPN Reward - EXPLOIT", exploit_res["cum reward"], exploit_step)
            writer.add_scalar("survive time - EXPLOIT", exploit_res["survive time"], exploit_step)   
            writer.add_scalar("Do action - EXPLOIT", exploit_res["Do action"], exploit_step)
            exploit_step+=1              

        else:
 #
            train_res=train_episode(Agents=Sub_Agents,MA_Controller=MultiAgent_Controll , Env=GlobalEnv,options=options)

            agents_counter+=train_res["agents counter"]
            if np.any(np.unique(np.array(agents_counter), return_counts=True)[1]>agents_kargs["start_training_capacity"]): start_exploit=True   # start exploit only when the training has started
            else:start_exploit=False
            writer.add_histogram("Buffer sizes", np.array(agents_counter), global_step=episode, bins=env_train.n_sub)
            writer.add_scalar("Manager buffer size", len(MultiAgent_Controll.buffer), episode)             

            writer.add_scalar("Cumulative L2RPN Reward", train_res["cum reward"], episode)
            writer.add_scalar("Survive time", train_res["survive time"], episode)

            if episode % hyperparameters["episodes for train"] == 0:
                MultiAgent_Controll.learn()
                for i in range(env_train.n_sub):
                    Sub_Agents[i].learn()

            if episode % agents_kargs["target_update_freq"]==0:
                if agents_kargs["checkpoints"]:
                    gp_manager.save_checkpoint(path=runs_folder)
                gp_manager.sync_target()
            if episode% MA_manager_kargs["target_update_freq"]==0:
                if agents_kargs["checkpoints"]:
                    MultiAgent_Controll.save_checkpoint()
                    MultiAgent_Controll.sync_target()




                for i in range(env_train.n_sub):
                    if agents_kargs["checkpoints"]:
                        Sub_Agents[i].save_estimator_checkpoint()
                    Sub_Agents[i].estimator_manager.sync_target()


        if hyperparameters["validation"]:
            if episode%hyperparameters["validation frequency"]==0:
                val_results=[]
                for chr_id in val_chr_ids:
                    val_res=exploit_episode(Agents=Sub_Agents,MA_Controller=MultiAgent_Controll, Env=GlobalEnv_val,options={"time serie id": chr_id})
                    writer.add_scalar("Cumulative L2RPN Reward - VAL", val_res["cum reward"], val_step)
                    writer.add_scalar("Survive time - VAL", val_res["survive time"], val_step)   
                    writer.add_scalar("Do action - VAL", val_res["Do action"], val_step)
                    val_results.append(val_res["survive time"])
                    val_step+=1   

                

                if not do_nothing_validation:
                    for chr_id in val_chr_ids:
                        do_n_res=do_nothing_episode(GlobalEnv_val, options={"time serie id": chr_id})
                        writer.add_scalar("Cumulative L2RPN Reward do nothing", do_n_res["cum reward"], do_nothing_step)
                        writer.add_scalar("Survive time do nothing", do_n_res["survive time"], do_nothing_step)
                        do_nothing_step+=1
                        do_nothing_validation=True

        if hyperparameters["test"] and test:
                for chr_id in test_chr_ids:
                    test_res=exploit_episode(Agents=Sub_Agents,MA_Controller=MultiAgent_Controll, Env=GlobalEnv_test,options={"time serie id": chr_id})
                    writer.add_scalar("Cumulative L2RPN Reward - TEST", test_res["cum reward"], test_step)
                    writer.add_scalar("Survive time - TEST", test_res["survive time"], test_step)   
                    writer.add_scalar("Do action - TEST", test_res["Do action"], test_step)
                    test_step+=1   
                    test=False

                if not do_nothing_test:
                    for chr_id in test_chr_ids:
                        do_n_res=do_nothing_episode(GlobalEnv_test, options={"time serie id": chr_id})
                        writer.add_scalar("Cumulative L2RPN Reward do nothing - TEST", do_n_res["cum reward"], do_nothing_step_test)
                        writer.add_scalar("Survive time do nothing - TEST", do_n_res["survive time"], do_nothing_step_test)
                        do_nothing_step_test+=1
                        do_nothing_validation=True

                

    if hyperparameters["test"]:
        for chr_id in test_chr_ids:
            test_res=exploit_episode(Agents=Sub_Agents,MA_Controller=MultiAgent_Controll, Env=GlobalEnv_test,options={"time serie id": chr_id})
            writer.add_scalar("Cumulative L2RPN Reward - TEST", test_res["cum reward"], test_step)
            writer.add_scalar("Survive time - TEST", test_res["survive time"], test_step)   
            writer.add_scalar("Do action - TEST", test_res["Do action"], test_step)
            test_step+=1   
        

    for i in range(init_obs.n_sub):
        Sub_Agents[i].close_writer()

    for i in range(init_obs.n_sub):
        Sub_Agents[i].save()

    MultiAgent_Controll.save_checkpoint()
    with open(runs_folder+"/Manager/buffer", 'wb') as f:
        pickle.dump(MultiAgent_Controll.buffer, f)

     

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment using JSON configuration.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Run main experiment
    main(config)
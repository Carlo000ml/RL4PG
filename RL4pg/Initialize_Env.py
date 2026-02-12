import grid2op
import argparse
import os
import shutil


def initialize_env(env_name):
    """
    Main function that processes the environment name and ensures all train, val, and test environments exist.

    Args:
        env_name (str): The base name of the environment to process.

    Returns:
        None
    """
    # Construct paths for the environments
    base_path = "/srv/mlg/home/cfab0002/data_grid2op/"+env_name   # os.path.join(os.getcwd(),"data_grid2op", env_name)
    print("the base path is - " , base_path , " - in case it is not the location of data_grid2op, change the file 'Initialize_Env.py' when computing the base_path " )
    train_env_path = base_path + "_train"
    val_env_path = base_path + "_val"
    test_env_path = base_path + "_test"

    # Check if all environments exist
    all_exist = all(os.path.exists(path) for path in [train_env_path, val_env_path, test_env_path])

    if all_exist:
        print(f"All environments exist: {train_env_path}, {val_env_path}, {test_env_path}")
        print("Skipping initialization.")
        return

    # If not all environments exist, delete any existing ones
    print("Some environments are missing. Reinitializing all environments...")
    for path in [train_env_path, val_env_path, test_env_path]:
        if os.path.exists(path):
            print(f"Deleting existing environment: {path}")
            shutil.rmtree(path)

    # Recreate environments
    print(f"Initializing environment: {env_name}")
    env = grid2op.make(env_name)
    _, _, _ = env.train_val_split_random(pct_val=1.0, pct_test=1.0, add_for_test="test")
    print("Environment successfully initialized.")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a script for a specified environment.")
    parser.add_argument("env_name", type=str, help="The name of the environment to process.")
    args = parser.parse_args()

    initialize_env(args.env_name)

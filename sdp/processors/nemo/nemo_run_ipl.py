# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import os
from pathlib import Path
from typing import Dict, List
import argparse
import nemo_run as run
from omegaconf import OmegaConf, open_dict

from sdp.utils import nemo_run_utils, ipl_utils
import logging
from sdp.processors.IPL.ipl_processors import TrainingCommandGenerator, InferenceCommandGenerator
# NEMO_ROOT = Path(__file__).absolute().parents[2]

def gather_mounts(cluster_cfg):
    """
    Gather all mounts from the cluster config including ones which are disjoint from the cluster_cfg.mounts list.
    It is used because Hydra does not support the ability to append to a list in the config file natively.

    Users can provide additional mounts from the command line using the following syntax:
    ++mount_<anything>='/src:/dest'

    Args:
        cluster_cfg: Cluster config dictionary with following fields.
            
            script (str): Path to the main Python script to be executed.
            script_config (str): Path to the YAML config used by the script.
            exp_name (str or None): Name of the experiment. If None, it is inferred from `exp_manager.name`
              in the script configuration.
            results_dir (str): Path to the directory where results should be saved.
            
            num_runs (int): Number of times to repeat the experiment.
            num_gpus (int): Number of GPUs to allocate per run.
            num_tasks_per_node (int): Number of tasks per node.
            max_runtime (str): Max allowed runtime in Slurm format (DD:HH:MM:SS). Default is "00:03:45:00".

            executor (str): Type of job executor, e.g., 'slurm', 'local'.

            ssh_tunnel:
                host (str): Hostname for the SSH tunnel.
                user (str): Username for SSH login. Can be `${USER}` to auto-resolve.
                job_dir (str): Remote path where jobs will be created and results uploaded.
                identity (str): Path to SSH identity file. Resolved from environment variable `${NEMO_OCI_IAD_SSH_IDENTITY}`.

            account (str): Account name used for SLURM job submissions.
            partition (str): Comma-separated list of SLURM partitions to use.
            job_name_prefix (str): Prefix for SLURM job names.

            containers:
                asr (str): URI or path to the container image used for ASR jobs.

            env_vars:
                List[str]: List of environment variable declarations to be set in the job,
                e.g., 'TOKENIZERS_PARALLELISM=false', 'HYDRA_FULL_ERROR=1', etc.
             
            required_env_vars (List[str]): List of env vars that **must** be present in the environment before running.
                - 'HF_TOKEN'
                - 'WANDB_KEY'
            mounts:
                - /paths/to/be/mounted:/paths/to/mount/t

            timeouts:
                partition_name: 04:00:00 (max runtime for execution)
    """ 
    # Gather all mounts from the cluster config including ones which are disjoint from the cluster_cfg.mounts list.
    mounts = cluster_cfg.get('mounts', [])
    # Resolve any mounts in th cluster config that need user expansion
    mounts = [os.path.expanduser(m) for m in mounts]

    keys = list(cluster_cfg.keys())
    # Check for any additional mounts in the cluster config
    with open_dict(cluster_cfg):
        for k in keys:
            if k.startswith("mount_"):  # Additional mount found
                logging.info(f"Found additional mount flag in the cluster config `{k}`. Adding it to the mounts list.")
                mounts.append(cluster_cfg[k])
                del cluster_cfg[k]  # Remove the key from the cluster config

        cluster_cfg['mounts'] = mounts
        logging.info(f"Final Mounts: {mounts}")


# def check_root_path(path, nemo_root):
#     """
#     Check if a path is in the NeMo root directory and convert it to a path that is relative to the NeMo root directory.
#     This is used to ensure that any path that is provided to this script will be in the NeMo root directory when
#     mounted in the container.

#     Args:
#         path: Path to check
#         nemo_root: NeMo root directory

#     Returns:
#         str: Path relative to the NeMo root directory
#     """
#     path = str(path)
#     nemo_root = str(nemo_root)

#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Path {path} does not exist.")

#     if not path.startswith(nemo_root):
#         raise ValueError(f"Path {path} is not in the NeMo root directory.")

#     new_path = path.replace(nemo_root, '/nemo_run/code/')
#     return new_path


def check_config_mount_paths(script_config, cluster_config):
    """
    Check if all path-like strings in the script config are mounted paths in the cluster config.
    If a path-like string is not a mounted path, raise an error.

    Args:
        script_config: Script config dictionary that represents the Model training/inference config
        cluster_config: Cluster config dictionary that represents the cluster configuration
    """
    # recursively walk all values of the script_config, checking if its a path-like string and if so, check if the path is a mounted path
    # if it is not, raise an error

    def filepath_check(v, cluster_cfg):
        if v.startswith(os.path.sep):  # check for absolute paths only
            logging.info(f"Checking if {v} is a mounted path")
            # Check if the path begins with mount path
            nemo_run_utils.check_if_mounted(cluster_cfg, v)

            # Check the file exists in the cluster at the unmounted path
            unmounted_path = nemo_run_utils.get_unmounted_filepath(cluster_cfg, v)
            nemo_run_utils.check_remote_mount_directories(unmounted_path, cluster_cfg)

    def check_mounted_path(cfg, cluster_cfg):
        if hasattr(cfg, 'items'):  # if the object is a dictionary
            for k, v in cfg.items():
                if hasattr(v, 'items'):  # if the value is a dictionary, recurse
                    check_mounted_path(v, cluster_cfg)

                elif isinstance(v, list):  # if the value is a list, check if its items are an absolute path
                    for item in v:
                        if isinstance(item, str):
                            filepath_check(item, cluster_cfg)

                elif isinstance(v, str):  # if the value is a string, check if its an absolute a path
                    filepath_check(v, cluster_cfg)

    check_mounted_path(script_config, cluster_config)

    return 


def get_export_variables_cmd(merged_cfg):
    wandb_key = os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB") or os.environ.get("WANDB_KEY", "")
    if not wandb_key:
        logging.warning("WANDB key not found in environment variables. WANDB logging will not work.")

        # Check if WANDB logging is enabled in the exp_manager config
        if merged_cfg.get('exp_manager', {}).get('create_wandb_logger', False):
            raise ValueError(
                "WANDB key is required for logging but was not found in environment variables. "
                "Please set WANDB_API_KEY to enable WANDB logging."
            )

    cmd = (
        "nvidia-smi && "
        "export PYTHONPATH=/nemo_run/code && "
        f"export HF_TOKEN={os.getenv('HF_TOKEN', '')} && "
        f"export WANDB_API_KEY={wandb_key} && ")
    
    return cmd

from sdp.processors.IPL.ipl_processors import TrainingCommandGenerator, InferenceCommandGenerator

def get_pseudo_labeling_command(
    train_command_config: dict, 
    inference_command_config: dict, 
    num_ipl_epochs: int,
    new_manifest_files,
    new_tarr_files,
    first_run: False,
    
) -> str:
    """
    Generate the pseudo-labeling command for the given configuration and training parameters using processors.

    Args:
        train_command_config (dict): Config for TrainingCommandGenerator.
        inference_command_config (dict): Config for InferenceCommandGenerator.
        num_ipl_epochs (int): Number of epochs to train with pseudo-labels.

    Returns:
        str: The constructed pseudo-labeling command.
    """
    # Instantiate processors
    train_proc = TrainingCommandGenerator(**train_command_config)
    infer_proc = InferenceCommandGenerator(**inference_command_config)

    exec_cmd = train_proc.process()
    exec_cmd += " && sleep 10"
    exec_cmd += " && " + infer_proc.process(first_run=first_run)

    # For subsequent epochs, set first_run to False
    for _ in range(num_ipl_epochs):
        exec_cmd += " && sleep 10"
        exec_cmd += " && " + train_proc.process(new_manifest_files, new_tarr_files)
        exec_cmd += " && " + infer_proc.process(first_run=False)

    return exec_cmd


def main(config_path: str):
    """
    Main entry point for running IPL training.
    
    Args:
        config_path (str): Path to the YAML configuration file
    """
    # Load the cluster config from YAML
    cluster_cfg = OmegaConf.load(config_path)
    
    # Process the required arguments from the cluster config
    script_path = cluster_cfg.script
    script_config_path = cluster_cfg.script_config
    results_dir = cluster_cfg.results_dir
    NEMO_ROOT = cluster_cfg.nemo_directory

    script_config_path = Path(script_config_path).absolute()

    # Gather all mounts from the cluster config
    gather_mounts(cluster_cfg)

    # Add the results directory to the cluster config as a mount path
    nemo_run_utils.add_mount_path(results_dir, '/results', cluster_cfg)

    # Create results and logdir
    log_dir = cluster_cfg.get('log_dir', os.path.join(results_dir, 'logs'))
    nemo_run_utils.create_remote_directory([results_dir, log_dir], cluster_cfg)

    # Load the script config
    script_config = OmegaConf.load(script_config_path)

    # Update the exp_manager runtime with the max_runtime from the cluster config
    import copy
    # Perform all path checks in the merged config
    if "ipl_training" in script_config.model:
        ipl_training = copy.deepcopy(script_config.model.ipl_training)
        # not to check the path
        del script_config.model.ipl_training.inference_config
    else:
        raise KeyError("Parameters for `IPL` training are not provided.")
    
    check_config_mount_paths(script_config, cluster_cfg)

    inference_config = ipl_training.inference_config
    inference_config_path = Path(inference_config).absolute()
    inference_config = OmegaConf.load(inference_config_path)

    # Resolve experiment name; if not provided in the script config file, check the cluster config
    exp_name = cluster_cfg.exp_name
    if exp_name is None:
        if 'exp_manager' in script_config and 'name' in script_config['exp_manager']:
            exp_name = script_config['exp_manager']['name']
        else:
            raise ValueError(
                "Experiment name not provided in the run config file (`exp_name`)) or the cluster config (inside exp_manager.name)"
            )

    # Begin NeMo Run setup
    with run.Experiment(exp_name) as exp:
        # Create the config file name
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        config_name = f"{exp_name}_{timestamp}_config.yaml"

        # Copy the merged config file to remote location's /results/configs directory
        config_dir = os.path.join(results_dir, 'configs')
        train_config_cluster = nemo_run_utils.create_remote_config(script_config, config_name, config_dir, cluster_cfg)

        # Prepare arguments for the slurm job
        job_name = f"{exp_name}_job"

        # Get run parameters from the config
        num_runs = cluster_cfg.num_runs  # Number of dependent jobs for this script
        num_gpus = cluster_cfg.get('num_gpus', script_config['trainer']['devices'])
        if isinstance(num_gpus, list):
            num_gpus = len(num_gpus)
        if num_gpus == -1:
            num_gpus = 1 if cluster_cfg['executor'] == 'local' else 8
            logging.warning(f"\n\nSetting num_gpus to {num_gpus} as it was set to -1\n\n")
        num_nodes = cluster_cfg.get('num_nodes', script_config['trainer'].get('num_nodes', 1))


        checkpoint_dir = os.path.join(
            os.path.join(script_config.exp_manager.exp_dir, script_config.exp_manager.name), "checkpoints"
        )
        checkpoint_name = os.path.join(checkpoint_dir, script_config.exp_manager.name + ".nemo")
        inference_config_paths, manifests, tarr_paths = nemo_run_utils.create_remote_inference_config(
            cluster_cfg, config_dir, inference_config, checkpoint_name
        )
        check_config_mount_paths(inference_config, cluster_cfg)

        train_command_generator_config = { 
            "nemo_directory": NEMO_ROOT,
            "training_config_local": script_config,
            "training_config_cluster": train_config_cluster,
            "training_script_path": script_path,
            "output_manifest_file": "./train_output_manifest_filepath.json",
        }   
        inference_command_generator_config = {
            "nemo_directory": NEMO_ROOT,
            "inference_config_paths": inference_config_paths,
            "manifests": manifests,
            "p_cache": script_config.model.ipl_training.p_cache,
            "num_gpus": num_nodes * num_gpus,
            "is_tarred": getattr(script_config.model.train_ds, "is_tarred", False),
            "output_manifest_file": "./inference_output_manifest_filepath.json",
        }


        cmd = get_pseudo_labeling_command(
            train_command_generator_config,
            inference_command_generator_config,
            num_ipl_epochs=script_config.model.ipl_training.num_ipl_epochs,
            new_manifest_files=manifests,
            new_tarr_files=tarr_paths,
            first_run=True,
        ) 

        # # Cast the cluster config to a dictionary for compatibility with NeMo Run
        cluster_cfg = OmegaConf.to_object(cluster_cfg)

       # logging.info(f"Scheduling {num_runs} runs of the script {script_path}...")

        task = None
        for run_id in range(num_runs):
            # Add the task to the experiment
            if run_id == 0:
                task = None
            else:
                if ipl_training:
                    cmd = get_pseudo_labeling_command(
                        train_command_generator_config,
                        inference_command_generator_config,
                        num_ipl_epochs=script_config.model.ipl_training.num_ipl_epochs,
                        new_manifest_files=manifests,
                        new_tarr_files=tarr_paths,
                        first_run=False
                    ) 
                task = [task]
            print(f"will add task")
            task = nemo_run_utils.add_task(
                exp,
                cmd=cmd,
                task_name=job_name,
                cluster_config=cluster_cfg,
                container=cluster_cfg['containers']['asr'],
                num_tasks=cluster_cfg.get('num_tasks', cluster_cfg.get('num_tasks_per_node', 1)),
                num_gpus=num_gpus,
                num_nodes=num_nodes,
                log_dir=nemo_run_utils.get_mounted_filepath(cluster_cfg, log_dir),
                partition=cluster_cfg.get('partition', None),
                task_dependencies=task,
            )

        # Run the experiment on the cluster with all the tasks
        nemo_run_utils.run_exp(exp, cluster_cfg)


if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser(description='Run IPL training with configuration')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file')
    args = parser.parse_args()
    
    main(args.config)

from sdp.processors.base_processor import BaseProcessor
from sdp.processors.IPL.ipl_processors import TrainingCommandGenerator, InferenceCommandGenerator
from omegaconf import OmegaConf, open_dict
import os
from pathlib import Path
import logging
import datetime
import nemo_run as run
from sdp.utils import nemo_run_utils

class NemoRunIPLProcessor(BaseProcessor):
    """
    A processor that handles Iterative Pseudo-Labeling (IPL) training workflow.
    
    Args:
        config_path (str): Path to the YAML configuration file containing IPL settings
        output_manifest_file (str): Path where the output manifest file will be written
        input_manifest_file (str, optional): Path to the input manifest file
    """
    
    def __init__(
        self,
        config_path: str,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.config_path = config_path
        
    def process(self):
        """
        Main processing method that implements the IPL workflow.
        This method:
        1. Loads and validates configurations
        2. Sets up training and inference command generators
        3. Executes the IPL training pipeline
        """
        # Load the cluster config from YAML
        cluster_cfg = OmegaConf.load(self.config_path)
        
        # Process the required arguments from the cluster config
        script_path = cluster_cfg.script
        script_config_path = cluster_cfg.script_config
        results_dir = cluster_cfg.results_dir
        nemo_root = cluster_cfg.nemo_directory
        inference_config = cluster_cfg.inference_config
        do_average = cluster_cfg.get('do_average', False)
        inference_config_path = Path(inference_config).absolute()

        inference_config = OmegaConf.load(inference_config_path)

        script_config_path = Path(script_config_path).absolute()

        # Gather all mounts from the cluster config
        self.gather_mounts(cluster_cfg)

        # Add the results directory to the cluster config as a mount path
        nemo_run_utils.add_mount_path(results_dir, '/results', cluster_cfg)

        # Create results and logdir
        log_dir = cluster_cfg.get('log_dir', os.path.join(results_dir, 'logs'))
        nemo_run_utils.create_remote_directory([results_dir, log_dir], cluster_cfg)

        # Load the script config
        script_config = OmegaConf.load(script_config_path)

        # Validate IPL training configuration
        if "ipl_training" not in script_config.model:
            raise KeyError("Parameters for `IPL` training are not provided.")
        # Check all paths in configs are properly mounted
   
        self.check_config_mount_paths(script_config, cluster_cfg)
        # Resolve experiment name
        exp_name = cluster_cfg.exp_name
        if exp_name is None:
            if 'exp_manager' in script_config and 'name' in script_config['exp_manager']:
                exp_name = script_config['exp_manager']['name']
            else:
                raise ValueError(
                    "Experiment name not provided in the run config file (`exp_name`) or the cluster config (inside exp_manager.name)"
                )

        # Begin NeMo Run setup
        with run.Experiment(exp_name) as exp:
            # Create the config file name
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            config_name = f"{exp_name}_{timestamp}_config.yaml"

            # Copy the merged config file to remote location's /results/configs directory
            config_dir = os.path.join(results_dir, 'configs')
            train_config_cluster = nemo_run_utils.create_remote_config(script_config, config_name, config_dir, cluster_cfg)

            # Get run parameters from the config
            num_runs = cluster_cfg.num_runs
            num_gpus = cluster_cfg.get('num_gpus', script_config['trainer']['devices'])
            if isinstance(num_gpus, list):
                num_gpus = len(num_gpus)
            if num_gpus == -1:
                num_gpus = 1 if cluster_cfg['executor'] == 'local' else 8
                logging.warning(f"\n\nSetting num_gpus to {num_gpus} as it was set to -1\n\n")
            num_nodes = cluster_cfg.get('num_nodes', script_config['trainer'].get('num_nodes', 1))

            # Set up checkpoint paths
            checkpoint_dir = os.path.join(
                os.path.join(script_config.exp_manager.exp_dir, script_config.exp_manager.name), "checkpoints"
            )
            checkpoint_name = os.path.join(checkpoint_dir, script_config.exp_manager.name + ".nemo")
            
            # Create remote inference config
            if do_average:
                avg_cmd, averaged_checkpoint = self.average_checkpoints(checkpoint_name, nemo_root)
            else:
                avg_cmd = None
                averaged_checkpoint = checkpoint_name
            inference_config_paths, manifests, tarr_paths = nemo_run_utils.create_remote_inference_config(
                cluster_cfg, config_dir, inference_config, averaged_checkpoint
            )
            self.check_config_mount_paths(inference_config, cluster_cfg)
            # Configure command generators
            train_command_generator_config = { 
                "nemo_directory": nemo_root,
                "training_config_local": script_config,
                "training_config_cluster": train_config_cluster,
                "training_script_path": script_path,
                "output_manifest_file": "./train_output_manifest_filepath.json",
            }   
            inference_command_generator_config = {
                "nemo_directory": nemo_root,
                "inference_config_paths": inference_config_paths,
                "manifests": manifests,
                "p_cache": cluster_cfg.p_cache,
                "num_gpus": num_nodes * num_gpus,
                "is_tarred": getattr(script_config.model.train_ds, "is_tarred", False),
                "output_manifest_file": "./inference_output_manifest_filepath.json",
            }

            # Generate the complete IPL command
            cmd = self.get_pseudo_labeling_command(
                train_command_generator_config,
                inference_command_generator_config,
                num_ipl_epochs=cluster_cfg.num_ipl_epochs,
                new_manifest_files=manifests,
                new_tarr_files=tarr_paths,
                first_run=True,
                avg_cmd=avg_cmd
            )

            # Cast the cluster config to a dictionary for compatibility with NeMo Run
            cluster_cfg = OmegaConf.to_object(cluster_cfg)

            # Schedule tasks
            task = None
            for run_id in range(num_runs):
                if run_id == 0:
                    task = None
                else:
                    cmd = self.get_pseudo_labeling_command(
                        train_command_generator_config,
                        inference_command_generator_config,
                        num_ipl_epochs=cluster_cfg.num_ipl_epochs,
                        new_manifest_files=manifests,
                        new_tarr_files=tarr_paths,
                        first_run=False
                    )
                    task = [task]

                task = nemo_run_utils.add_task(
                    exp,
                    cmd=cmd,
                    task_name=f"{exp_name}_job",
                    cluster_config=cluster_cfg,
                    container=cluster_cfg['containers']['asr'],
                    num_tasks=cluster_cfg.get('num_tasks', cluster_cfg.get('num_tasks_per_node', 1)),
                    num_gpus=num_gpus,
                    num_nodes=num_nodes,
                    log_dir=nemo_run_utils.get_mounted_filepath(cluster_cfg, log_dir),
                    partition=cluster_cfg.get('partition', None),
                    task_dependencies=task,
                )

            # Run the experiment
            nemo_run_utils.run_exp(exp, cluster_cfg)

    def gather_mounts(self, cluster_cfg):
        """
        Gather all mounts from the cluster config including ones which are disjoint from the cluster_cfg.mounts list.
        
        Args:
            cluster_cfg: Cluster config dictionary
        """
        mounts = cluster_cfg.get('mounts', [])
        mounts = [os.path.expanduser(m) for m in mounts]

        keys = list(cluster_cfg.keys())
        with open_dict(cluster_cfg):
            for k in keys:
                if k.startswith("mount_"):
                    logging.info(f"Found additional mount flag in the cluster config `{k}`. Adding it to the mounts list.")
                    mounts.append(cluster_cfg[k])
                    del cluster_cfg[k]

            cluster_cfg['mounts'] = mounts
            logging.info(f"Final Mounts: {mounts}")

    def check_config_mount_paths(self, script_config, cluster_config):
        """
        Check if all path-like strings in the script config are mounted paths in the cluster config.
        
        Args:
            script_config: Script config dictionary
            cluster_config: Cluster config dictionary
        """
        def filepath_check(v, cluster_cfg):
            if v.startswith(os.path.sep):
                logging.info(f"Checking if {v} is a mounted path")
                nemo_run_utils.check_if_mounted(cluster_cfg, v)
                unmounted_path = nemo_run_utils.get_unmounted_filepath(cluster_cfg, v)
                nemo_run_utils.check_remote_mount_directories(unmounted_path, cluster_cfg)

        def check_mounted_path(cfg, cluster_cfg):
            if hasattr(cfg, 'items'):
                for k, v in cfg.items():
                    if hasattr(v, 'items'):
                        check_mounted_path(v, cluster_cfg)
                    elif isinstance(v, list):
                        for item in v:
                            if isinstance(item, str):
                                filepath_check(item, cluster_cfg)
                    elif isinstance(v, str):
                        filepath_check(v, cluster_cfg)

        check_mounted_path(script_config, cluster_config)

    def get_pseudo_labeling_command(
        self,
        train_command_config: dict,
        inference_command_config: dict,
        num_ipl_epochs: int,
        new_manifest_files,
        new_tarr_files,
        first_run: bool = False,
        avg_cmd: str = None
    ) -> str:
        """
        Generate the pseudo-labeling command for the given configuration and training parameters.

        Args:
            train_command_config (dict): Config for TrainingCommandGenerator
            inference_command_config (dict): Config for InferenceCommandGenerator
            num_ipl_epochs (int): Number of epochs to train with pseudo-labels
            new_manifest_files: List of manifest files to use
            new_tarr_files: List of tarred audio files to use
            first_run (bool): Whether this is the first run of pseudo-labeling

        Returns:
            str: The constructed pseudo-labeling command
        """
        train_proc = TrainingCommandGenerator(**train_command_config)
        infer_proc = InferenceCommandGenerator(**inference_command_config)

        exec_cmd = self.get_export_variables_cmd(train_command_config["training_config_local"], train_command_config["nemo_directory"])
        exec_cmd += train_proc.process()
        exec_cmd += " && sleep 10"
        if avg_cmd:
            exec_cmd += " && " + avg_cmd
        exec_cmd += " && " + infer_proc.process(first_run=first_run)

        for _ in range(num_ipl_epochs):
            exec_cmd += " && sleep 10"
            exec_cmd += " && " + train_proc.process(new_manifest_files, new_tarr_files)
            if avg_cmd:
                exec_cmd += " && " + avg_cmd
            exec_cmd += " " + infer_proc.process(first_run=False)

        return exec_cmd

    def get_export_variables_cmd(self, merged_cfg , nemo_root):
        """Generate command to export required environment variables."""
        wandb_key = os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB") or os.environ.get("WANDB_KEY", "")
        if not wandb_key:
            logging.warning("WANDB key not found in environment variables. WANDB logging will not work.")

            if merged_cfg.get('exp_manager', {}).get('create_wandb_logger', False):
                raise ValueError(
                    "WANDB key is required for logging but was not found in environment variables. "
                    "Please set WANDB_API_KEY to enable WANDB logging."
                )

        cmd = (
            "nvidia-smi && "
            f"export PYTHONPATH={nemo_root} && "
            f"export HF_TOKEN={os.getenv('HF_TOKEN', '')} && "
            f"export WANDB_API_KEY={wandb_key} && ")
        
        return cmd
    
    def average_checkpoints(self, checkpoint_path: str, nemo_root:str) -> str:
        """
        Generates the command to average all checkpoints in the given directory and returns the path to the averaged checkpoint.
        
        Args:
            checkpoint_path (str): Path to the directory containing checkpoints
            
        Returns:
            tuple: (command to run, path to the averaged checkpoint file)
        """
        # Get the directory containing the checkpoints
        checkpoint_dir = os.path.dirname(checkpoint_path)
        
        # Construct the command for checkpoint averaging
        cmd = f"python {nemo_root}/scripts/checkpoint_averaging/legacy/checkpoint_averaging.py {checkpoint_dir}"
        
        # The averaged checkpoint will have the same name but with '-averaged' suffix
        checkpoint_name = os.path.basename(checkpoint_path)
        base_name = os.path.splitext(checkpoint_name)[0]
        averaged_checkpoint = os.path.join(checkpoint_dir, f"{base_name}-averaged.nemo")
        
        return cmd, averaged_checkpoint

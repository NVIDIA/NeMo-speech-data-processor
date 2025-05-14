# Standard library imports
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Third-party imports
from omegaconf import DictConfig, OmegaConf, open_dict
import logging
import json
# Local imports
from sdp.processors.base_processor import BaseProcessor


class TrainingCommandGenerator(BaseProcessor):
    """
    A processor that generates training commands for NeMo models with support for both local and cluster configurations.
    Handles manifest file updates and tarred audio filepath management for training datasets.

    Args:
        training_config_local (str): Path to the local machine configuration file
        training_config_cluster (str): Path to the cluster configuration file
        training_script_path (str): Path to the training script relative to nemo_directory
        nemo_directory (str): Base directory for NeMo framework
        new_manifest_files (str, optional): New manifest files to add to the training configuration
        new_tarred_audio_filepaths (str, optional): New tarred audio filepaths to add to the training configuration
        **kwargs: Additional arguments passed to the parent BaseProcessor class
    """

    def __init__(
        self,
        training_config_local: str,      # Local machine config path
        training_config_cluster: str,    # Cluster config path
        training_script_path: str,       # Path to training script
        nemo_directory: str,             # Base directory for NeMo
        new_manifest_files: str = None,  # New manifest files to add
        new_tarred_audio_filepaths: str = None,  # New tarred audio paths
        **kwargs
    ):
        super().__init__(**kwargs)

        # Paths on the current machine
        self.training_config_local = OmegaConf.load(training_config_local)
        self.training_config_cluster = training_config_cluster
        self.training_script_path = os.path.join(nemo_directory, training_script_path)
        self.nemo_directory = nemo_directory
        self.new_manifest_files = new_manifest_files
        self.new_tarred_audio_filepaths = new_tarred_audio_filepaths

    def process(self) -> str:
        """
        Generates the training command based on the processor's configuration.
        If new manifest files are provided, updates the training configuration accordingly.

        Returns:
            str: The complete training command to be executed on the cluster
        """
        
        if self.new_manifest_files is None:
            cmd = self.get_execution_script(
                cluster_script_path=self.training_script_path,
                local_config=self.training_config_local,
                cluster_config_path=self.training_config_cluster
            )
        else:
            updated_manifest_filepaths, updated_tarred_audio_filepaths = self.update_training_sets(
                config=self.training_config_local,
                updated_manifest_filepaths=self.new_manifest_files,
                updated_tarred_audio_filepaths=self.new_tarred_audio_filepaths
            )
            cmd = self.get_execution_script(
                cluster_script_path=self.training_script_path,
                local_config=self.training_config_local,
                cluster_config_path=self.training_config_cluster,
                updated_manifest_filepaths=updated_manifest_filepaths,
                updated_tarred_filepaths=updated_tarred_audio_filepaths
            )
        return cmd

    def get_execution_script(
        self,
        cluster_script_path: str,
        local_config: DictConfig,
        cluster_config_path: str,
        updated_manifest_filepaths: Optional[str] = None,
        updated_tarred_filepaths: Optional[str] = None
    ) -> str:
        """
        Create the command to run the script on the cluster.

        Args:
            cluster_script_path (str): Path to the script to run on the cluster
            local_config (DictConfig): Local configuration loaded from training_config_local
            cluster_config_path (str): Path to the cluster configuration file
            updated_manifest_filepaths (str, optional): Path to the updated manifest file
            updated_tarred_filepaths (str, optional): Path to the updated tarred audio filepaths

        Returns:
            str: Command to run the script on the cluster
        """
        # Get the WANDB API key from the environment variables
        wandb_key = os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB") or os.environ.get("WANDB_KEY", "")
        if not wandb_key:
            logging.warning("WANDB key not found in environment variables. WANDB logging will not work.")

            # Check if WANDB logging is enabled in the exp_manager config
            if local_config.get('exp_manager', {}).get('create_wandb_logger', False):
                raise ValueError(
                    "WANDB key is required for logging but was not found in environment variables. "
                    "Please set WANDB_API_KEY to enable WANDB logging."
                )

        # Prepare the base command
        config_path = os.path.dirname(cluster_config_path)
        config_name = os.path.basename(cluster_config_path)
        cmd = (
            "nvidia-smi && "
            f"cd {os.path.dirname(cluster_script_path)} && "
            f"python -u -B {os.path.basename(cluster_script_path)} "
            f"--config-path {config_path} --config-name \"{config_name}\""
        )

        # Add additional parameters if provided
        if updated_manifest_filepaths:
            cmd += f" model.train_ds.manifest_filepath={updated_manifest_filepaths}"
        if updated_tarred_filepaths:
            cmd += f" model.train_ds.tarred_audio_filepaths={updated_tarred_filepaths}"
        output_data = {"training_command": cmd}
        with open(self.output_manifest_file, 'w') as f:
            json.dump(output_data, f, indent=4)
        return cmd
    
    def get_transcribed_names(self, manifest_filepaths: List[str], is_tarred: bool=False) -> List[List[str]]:
        """
        Generates a list of modified file paths by prepending 'transcribed_' to the filenames.
        The use case is for non AIStore datasets

        Args:
            manifest_filepaths (list of str): A list of file paths to be modified.

        Returns:
            list of list of str: A list where each element is a single-item list containing the updated file path.
        Example:
            >>> manifest_filepaths = [
            ...     "/path/to/manifest_1.json",
            ...     "/path/to/manifest_2.json"
            ... ]
            >>> get_transcribed_names(manifest_filepaths)
            [
                ["/path/to/prefix_transcribed_manifest_1.json"],
                ["/path/to/prefix_transcribed_manifest_2.json"]
            ]
        """
        # For manifest_filepath, modify the filenames by prepending 'prefix_transcribed_'
        transcribed_paths = []

        for file_path in manifest_filepaths:
            directory, filename = os.path.split(file_path)
            
            new_filename = (
                f"transcribed_{filename}" if is_tarred 
                else f"transcribed_manifest.json"
            )
            transcribed_paths.append([os.path.join(directory, new_filename)])

        return transcribed_paths

    def update_training_sets(
        self,
        config: DictConfig,
        updated_manifest_filepaths: List[str],
        updated_tarred_audio_filepaths: Optional[List[str]] = None
    ) -> Tuple[str, str]:
        """
        Updates the training dataset configuration by adding pseudo-labeled datasets
        to the training paths based on the dataset type.

        Args:
            config (DictConfig): Training config file to be updated
            updated_manifest_filepaths (List[str]): List of updated manifest file paths to be included
            updated_tarred_audio_filepaths (Optional[List[str]]): List of updated tarred audio filepaths to be included

        Returns:
            Tuple[str, str]: A tuple containing:
                - Updated manifest file paths as a string, formatted for Omegaconf
                - Updated tarred audio file paths as a string, formatted for Omegaconf
        """
        print(f"updated_manifest_filepaths {updated_manifest_filepaths}")
        updated_manifest_filepaths = self.get_transcribed_names(updated_manifest_filepaths,is_tarred=config.model.train_ds.get("is_tarred", False))
        manifest_filepath = config.model.train_ds.manifest_filepath
        if updated_tarred_audio_filepaths:
            updated_tarred_audio_filepaths = [[path] for path in updated_tarred_audio_filepaths]

        # Updating the configuration based on dataset types
        if config.model.train_ds.get("is_tarred", False):
            tarred_audio_filepaths = config.model.train_ds.tarred_audio_filepaths
            if isinstance(tarred_audio_filepaths, str):
                updated_tarred_audio_filepaths.append([tarred_audio_filepaths])
                updated_manifest_filepaths.append([manifest_filepath])
            else:
                updated_tarred_audio_filepaths += tarred_audio_filepaths
                updated_manifest_filepaths += manifest_filepath
        else:
            print(f"config.model.train_ds.get {config.model.train_ds.get('use_lhotse')}")
            if config.model.train_ds.get("use_lhotse", False):
                if isinstance(manifest_filepath, str):
                    updated_manifest_filepaths.append([manifest_filepath])
                else:
                    updated_manifest_filepaths += manifest_filepath
            else:
                updated_manifest_filepaths = [item for sublist in updated_manifest_filepaths for item in sublist]
                if isinstance(manifest_filepath, str):
                    updated_manifest_filepaths.append(manifest_filepath)
                else:
                    updated_manifest_filepaths += manifest_filepath

        # Returning strings formatted for Omegaconf
        return (
            str(updated_manifest_filepaths).replace(", ", ","),
            str(updated_tarred_audio_filepaths).replace(", ", ",") if updated_tarred_audio_filepaths else None,
        )


class InferenceCommandGenerator(BaseProcessor):
    """
    A processor that generates inference commands for pseudo-labeling.

    Args:
        nemo_directory (str): Base directory for NeMo framework
        inference_local_config (str): Path to the local configuration file
        inference_config_paths (str): Path to the inference configuration files
        manifests (str): Path to the manifest files
        p_cache (float): What part of pseudo-labels to update
        num_gpus (int): Number of GPUs to use
        is_tarred (bool): Whether the audio is tarred
        first_run (bool): Whether this is the first run of pseudo-labeling
        **kwargs: Additional arguments passed to the parent BaseProcessor class
    """

    def __init__(
        self,
        nemo_directory: str, 
        inference_config_paths: str,
        manifests:  str,
        p_cache: float,
        num_gpus, int,
        is_tarred: bool = False,
        first_run: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)

        # Paths on the current machine
        self.inference_config_paths = inference_config_paths
        self.nemo_directory = nemo_directory
        self.inference_script_path = os.path.join(nemo_directory, "examples/asr/transcribe_speech_parallel.py")
        self.first_run = first_run  
        self.manifests = manifests
        self.p_cache = p_cache
        self.num_gpus = num_gpus
        self.is_tarred = is_tarred

    def process(self): 
        """
        Generate the pseudo-labeling command for the given configuration and training parameters.

        Args:
            merged_config (Dict): Merged configuration containing model and dataset settings.
            config_name (str): Name of the configuration file to be used.
            cluster_script_path (str): Path to the cluster execution script.
            config_dir (str): Directory containing the configuration files.
            ipl_training (Dict[str, any]): Dictionary containing:
                - first_run (bool): Whether this is the first run of pseudo-labeling.
                - num_gpus (int): Number of GPUs to use.
                - inference_config_paths (List[str]): List of inference configuration file paths.
                - manifests (List[str]): List of manifest file paths.
                - tarr_paths (List[str]): List of tarred audio file paths.
                - num_ipl_epochs (int): Number of epochs to train with pseudo-labels.
                - p_cache (float): What part of pseudo-labels to update.

        Returns:
            str: The constructed pseudo-labeling command.
        """
        cmd = ""
        prediction_directories_str = " ".join([os.path.dirname(path) for path in self.manifests])
        inference_config_paths_str = " ".join(self.inference_config_paths)        
        write_transcription_path = os.path.join(self.nemo_directory, "scripts/pseudo_labeling/write_transcribed_files.py")
        update_inference_config_path = os.path.join(self.nemo_directory, "scripts/pseudo_labeling/update_inference_config.pys")
        if self.first_run:
            cmd += f"{self.get_pl_inference_command(self.inference_config_paths, shuffle=False)}"
            cmd += (
                f" && python {write_transcription_path} "
                f"--prediction_filepaths {prediction_directories_str} --full_pass"
            )
            if self.is_tarred:
                cmd += " --is_tarred"
            cmd += (
                f" && python {update_inference_config_path} "
                f"--inference_configs {inference_config_paths_str} --p_cache {self.p_cache} --num_gpus {self.num_gpus}"
            )

       
        cmd += f" && {self.get_pl_inference_command(self.inference_config_paths, shuffle=True)}"
        cmd += (
            f" && python {write_transcription_path} "
            f"--prediction_filepaths {prediction_directories_str} "
        )
        if self.is_tarred:
            cmd += " --is_tarred"

        output_data = {"inference_command": cmd}
        with open(self.output_manifest_file, 'w') as f:
            json.dump(output_data, f, indent=4)

        return cmd


    def get_pl_inference_command(self, inference_configs, shuffle=None):
        """
        Generate a command to run PL inference with multiple configuration files.
        Args:
            inference_configs (list): List of configuration file paths.
            shuffle (bool, optional): Whether to enable shuffling in predict_ds.

        Returns:
            str: Combined command string to execute PL inference.
        """
        cmd_list = []
        for config in inference_configs:
            config_path = os.path.dirname(config)
            config_name = os.path.basename(config)
            cmd = f"python {self.inference_script_path} --config-path {config_path} --config-name {config_name}"
            if shuffle is not None:
                cmd += f" predict_ds.shuffle={shuffle}"
            cmd_list.append(cmd)

        return " && ".join(cmd_list)
    
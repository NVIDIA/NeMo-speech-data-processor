
import os
import subprocess
from pathlib import Path
from typing import Optional
from typing import Dict, List
from omegaconf import OmegaConf, open_dict
from nemo.utils import logging
from sdp.processors.base_processor import BaseProcessor


class IPLCommandGenerator(BaseProcessor):
    """This processor performs ASR inference on each utterance of the input manifest.

    ASR predictions will be saved in the ``pred_text`` key.

    Args:
        pretrained_model (str): the name or the filepath of the pretrained NeMo ASR model
            which will be used to do inference.
        batch_size (int): the batch size to use for ASR inference. Defaults to 32.

    Returns:
         The same data as in the input manifest with an additional field
         ``pred_text`` containing ASR model's predictions.
    """

    def __init__(
        self,
        training_config: str,
        infenrece_config: str,
        training_script_path: str,
        nemo_directory: str,
        num_ipl_epochs: 50,

        **kwargs
    ):
        super().__init__(**kwargs)
        # Paths on the current machine
        self.training_config = OmegaConf.load(training_config)
        self.infenrece_config = OmegaConf.load(infenrece_config) 
        self.training_script_path = os.path.join(nemo_directory, training_script_path)
        self.nemo_directory = nemo_directory
        self.num_ipl_epochs = num_ipl_epochs

    def process(self):
        """.""" 
        


        

    def get_training_script_cmd(self, cluster_script_path, config_name, updated_manifest_filepaths=None, updated_tarred_filepaths=None):
        """
        Create the command to run the script on the cluster.

        Args:
            cluster_script_path (str): Path to the script to run on the cluster.
            config_name (str): Name of the config file to use for the script.
            updated_manifest_filepaths (str, optional): Path to the updated manifest file. Defaults to None.
            updated_tarred_filepaths (str, optional): Path to the updated tarred audio filepaths. Defaults to None.

        Returns:
            str: Command to run the script on the cluster.
        """

        # Prepare the base command for training
        cmd = (
            "find /results/ -name '*-unfinished' -type f -delete && "
            f"cd {os.path.dirname(cluster_script_path)} && "
            f"python -u -B {os.path.basename(cluster_script_path)} "
            f"--config-path \"/results/configs\" --config-name \"{config_name}\""
        )

        # Add additional parameters if provided
        if updated_manifest_filepaths:
            cmd += f" model.train_ds.manifest_filepath={updated_manifest_filepaths}"
        if updated_tarred_filepaths:
            cmd += f" model.train_ds.tarred_audio_filepaths={updated_tarred_filepaths}"

        return cmd

    def get_export_variables_cmd(self, merged_cfg):
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

    def get_pl_inference_command(self, inference_configs, shuffle=None):
        """
        Generate a command to run PL inference with multiple configuration files.
        Args:
            inference_configs (list): List of configuration file paths.

        Returns:
            str: Combined command string to execute PL inference.
        """
        # Base command template

        base_cmd = "python /nemo_run/code/examples/asr/transcribe_speech_parallel.py --config-path \"/results/configs\" --config-name {config_name}"
        if shuffle is not None:
            base_cmd += f" predict_ds.shuffle={shuffle}"

        # Generate the command list
        cmd_list = [base_cmd.format(config_name=os.path.basename(config)) for config in inference_configs]

        # Combine the commands with " && " separator
        return " && ".join(cmd_list)

    def get_pseudo_labeling_command(
        self, merged_config: Dict, config_name: str, cluster_script_path: str, config_dir: str, ipl_training: Dict[str, any]) -> str:
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
        
        prediction_directories_str = " ".join([os.path.dirname(path) for path in ipl_training['manifests']])
        inference_config_paths_str = " ".join(ipl_training['inference_config_paths'])

        updated_manifest_filepaths, updated_tarred_audio_filepaths = ipl_utils.update_training_sets(
            merged_config, ipl_training["manifests"], ipl_training.get("tarr_paths", None), ipl_training["prefix"]
        )
        exec_cmd = self.get_export_variables_cmd(merged_cfg=merged_config)
        exec_cmd += self.get_training_script_cmd(cluster_script_path, config_name)
        exec_cmd += " && sleep 10"
        if ipl_training.get("first_run", False):
            exec_cmd += f" && {self.get_pl_inference_command(ipl_training['inference_config_paths'], shuffle=False)}"
            exec_cmd += (
                f" && python /nemo_run/code/examples/asr/run_write_transcribed_files.py "
                f"--prediction_filepaths {prediction_directories_str} --full_pass --prefix {ipl_training['prefix']}"
            )
            if merged_config.model.train_ds.is_tarred:
                exec_cmd += " --is_tarred"
            exec_cmd += (
                f" && python /nemo_run/code/examples/asr/run_update_inf_config.py "
                f"--inference_configs {inference_config_paths_str} --p_cache {ipl_training['p_cache']} --num_gpus {ipl_training['num_gpus']}"
            )

        # If run has been interupted user has to change `num_ipl_epochs` in the config
        for _ in range(ipl_training["num_ipl_epochs"]):
            run_script = self.get_training_script_cmd(
                cluster_script_path, config_name, updated_manifest_filepaths, updated_tarred_audio_filepaths
            )
            exec_cmd += " && sleep 10"
            exec_cmd += f" && {run_script}"
            exec_cmd += f" && {self.get_pl_inference_command(ipl_training['inference_config_paths'],shuffle=True)}"
            exec_cmd += (
                f" && python /nemo_run/code/examples/asr/run_write_transcribed_files.py "
                f"--prediction_filepaths {prediction_directories_str} "
                f"--prefix {ipl_training['prefix']}"
            )
            if merged_config.model.train_ds.is_tarred:
                exec_cmd += " --is_tarred"

        return exec_cmd
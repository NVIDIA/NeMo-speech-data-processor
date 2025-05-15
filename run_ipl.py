import copy
import glob
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict
import torch
from typing import List, Optional, Tuple, Union
from omegaconf import OmegaConf, open_dict
#import sdp.processors.nemo.ipl_utils as ipl_utils
#from nemo.core.config import hydra_runner
from sdp.processors.IPL.ipl_processors import TrainingCommandGenerator

# def check_training_finished(log_dir):
#     """
#     Searches to see ig lightning finished training .
#     Parameters:
#         log_dir (str): Directory where logs are stored.
#     """
#     print(f"************************************************")
#     print(f"************************************************")

#     if not os.path.exists(log_dir):
#         print(f"Log directory '{log_dir}' does not exist.")
#         return
#     print(f"")
#     log_pattern = os.path.join(log_dir, f"lightning_logs.txt")
#     command = f"grep -ri '`Trainer.fit` stopped:' {log_pattern}"

#     result = subprocess.run(command, shell=True, capture_output=True, text=True)
#     if result.stdout:
#         print("Stopping reasons found:")
#         print(result.stdout)
#         return True
#     else:
#         print("No stopping reasons found in the logs.")
#         return False
    
# def get_command_for_inference(
#     inference_config: str, inference_config_dir: Union[str, Path], p_cache: float, checkpoint: str, nemo_path: str
# ) -> Tuple[str, List[str], List[str]]:
#     """
#     Generates the command string for running speech inference with transcribe_speech_parallel.
#     Args:
#         inference_config (str): Path to the base inference configuration file.
#         inference_config_dir (Union[str, Path]): Directory to store temporary modified configurations.
#         p_cache (float): Proportion of the dataset to be cached for pseudo-labeling.
#         checkpoint (str): Path to the model checkpoint to use for inference.
#     Returns:
#         Tuple[str, List[str], List[str]]:
#             - The command string to execute inference for all specified manifests.
#             - List of output directories corresponding to each manifest.
#             - List of completed full pass transcribed manifest paths, if any.
#     """
#     """"""
    
#     manifests, tarr_audio_files = ipl_utils.separate_multiple_transcriptions(inference_config)
#     num_gpus = torch.cuda.device_count()
#     output_dirs = []
#     cmd = ""
#     for i in range(len(manifests)):
#         print()
#         print(f"manifests  {manifests[i]}")
#         output_dir = os.path.dirname(manifests[i])
#         output_dirs.append(output_dir)
#         print(f"output_dir {output_dir}")
#         base_cfg = OmegaConf.load(inference_config)
#         print(f"inference_config_dir {inference_config_dir}")
#         print()
#         temp_config_dir = Path(str(inference_config_dir) + "/temp_configs").absolute()
#         os.makedirs(temp_config_dir, exist_ok=True)
#         modified_cfg = copy.deepcopy(base_cfg)

#         # Check if we need to run inference on the whole set or update part of it
#         full_pass_done = glob.glob(os.path.join(output_dir, 'transcribed_manifest*'))
#         if full_pass_done:
#             number_of_files = ipl_utils.count_files_for_pseudo_labeling(manifests[i], bool(tarr_audio_files))
#             limit_predict_batches = int((number_of_files * p_cache) / (modified_cfg.predict_ds.batch_size * num_gpus))
#             OmegaConf.update(modified_cfg, "trainer.limit_predict_batches", limit_predict_batches)

#         # Replace OmegaConf updates with simple assignments
#         OmegaConf.update(modified_cfg, "output_path", output_dir)
#         OmegaConf.update(modified_cfg, "predict_ds.manifest_filepath", manifests[i])
#         if tarr_audio_files:
#             OmegaConf.update(modified_cfg, "predict_ds.tarred_audio_filepaths", tarr_audio_files[i])
#         OmegaConf.update(modified_cfg, "model", checkpoint)

#         temp_config_file = os.path.join(temp_config_dir, f"modified_config_{i}.yaml")
#         OmegaConf.save(modified_cfg, temp_config_file)
#         trancribe_script = nemo_path + "/" + "transcribe_speech_parallel.py"
#         cmd += f"python {trancribe_script} --config-path {temp_config_dir} --config-name modified_config_{i}.yaml && "

#     # Remove trailing '&&' from the final command string
#     cmd = cmd.rstrip(" &&")

#     print(f"Inference command: {cmd}")
#     return cmd, output_dirs, full_pass_done


# def merge_configs(script_config_path, run_config):
#     # Load the configurations
#     script_config = OmegaConf.load(script_config_path)

#     print(run_config)

#     # Keep track of the original keys in script_config
#     original_script_keys = set(script_config.keys())

#     # Merge only the 'training' part of run_config with script_config
#     result = OmegaConf.merge(script_config, run_config)

#     with open_dict(result):
#         for k in run_config.keys():
#             if k in result and k not in original_script_keys:
#                 del result[k]

#     def check_missing_values(cfg):
#         if hasattr(cfg, 'items'):
#             for k, v in cfg.items():
#                 if hasattr(v, 'items'):
#                     check_missing_values(v)
#                 elif v == '???':
#                     raise ValueError(f"Missing value for key {k} in the config file")

#     check_missing_values(result)
#     result.exp_manager.resume_if_exists = True
#     return result


# def get_execution_script(cluster_script_path: str, config_name: str, config_path: str, nemo_path: str) -> str:
#     """
#     Constructs a command string to execute a training with the specified configuration.
#     Args:
#         cluster_script_path (str): Path to the cluster script to be executed.
#         config_name (str): Name of the configuration file or object to be passed as a parameter.
#         config_path (str): Path to the directory where the configuration resides.
#     Returns:
#         str: A formatted command string ready for execution.
#     """
#     # Create the command to run the script
#     cluster_script_path = nemo_path + "/" + cluster_script_path
#     cmd = """
#         python {cluster_script_path} --config-path {config_path} --config-name "{config_name}" 
#     """
#     print("in get_execution_script")
#     print(f"cluster_script_path {cluster_script_path}")
#     format_dict = dict(
#         cluster_script_path=cluster_script_path,
#         config_path=config_path,
#         config_name=config_name,
#     )
#     cmd = cmd.format(**format_dict)
#     print(f"format cmd {cmd}")

#     return cmd


# def find_checkpoint_dir(base_path):
#     """
#     Find the 'checkpoints' folder in the directory structure.
#     Parameters:
#         base_path (str): The base directory path to search from.
#     """
#     for root, dirs, files in os.walk(base_path):
#         for dir_name in dirs:
#             if dir_name == "checkpoints":
#                 return os.path.join(root, dir_name), root
#     return None, None


def main():
    config = {
        "training_config_local": "/home/ntadevosyan/code/canary_ngpt/NeMo/ngpt_rnnt_bpe.yaml",
        "training_config_cluster": "path/to/your/cluster/config.yaml",
        "training_script_path": "path/to/training/script.py",
        "nemo_directory": "path/to/nemo/directory",
        "output_manifest_file": "path/to/output/manifest.json",
        "new_manifest_files": None,  # or list of manifest files if you have them
        "new_tarred_audio_filepaths": None  # or list of tarred audio paths if you have them
    }
    processor = TrainingCommandGenerator(**config)
    cmd = processor.process(param="str")
    print("Generated command:", cmd)

if __name__ == '__main__':
    main()

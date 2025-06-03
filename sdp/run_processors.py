# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import logging
import os
import tempfile
import uuid
from typing import List, Optional
import psutil
import json

import hydra
from omegaconf import OmegaConf, open_dict

from sdp.logging import logger

from sdp.utils.import_manager import ImportManager

# registering new resolvers to simplify config files
OmegaConf.register_new_resolver("subfield", lambda node, field: node[field])
OmegaConf.register_new_resolver("not", lambda x: not x)
OmegaConf.register_new_resolver("equal", lambda field, value: field == value)


# customizing logger
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    '[SDP %(levelname)1.1s %(asctime)s %(module)s:%(lineno)d] %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S",
)
handler.setFormatter(formatter)
logger.handlers
logger.addHandler(handler)
logger.propagate = False

def update_processor_imports(config_path: str, init_file: str = None):
    """
    Update processor imports based on config file.
    
    Args:
        config_path: Path to the YAML config file
        init_file: Optional path to __init__.py file to update
    """
    try:
        import yaml
        manager = ImportManager()
        manager.sync_with_config(config_path, init_file)
        logger.info(f"Successfully updated imports for config: {config_path}")
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML config: {e}")
    except ImportError as e:
        logger.error(f"Import error: {e}")
    except ValueError as e:  # For unexpected data structures in the YAML config
        logger.error(f"Invalid value encountered: {e}")
    except Exception as e:  # For any other unexpected errors
        logger.error(f"An unexpected error occurred: {e}")


def select_subset(input_list: List, select_str: str) -> List:
    """This function parses a string and selects objects based on that.

    The string is expected to be a valid representation of Python slice. The
    only difference with using an actual slice is that we are always returning
    a list, never a single element. See examples below for more details.

    Examples::

        >>> processors_to_run = [1, 2, 3, 4, 5]
        >>> select_subset(processors_to_run, "3:") # to exclude first 3 objects
        [4, 5]

        >>> select_subset(processors_to_run, ":-1") # to select all but last
        [1, 2, 3, 4]

        >>> select_subset(processors_to_run, "2:5") # to select 3rd to 5th
        [3, 4, 5]

        >>> # note that unlike normal slice, we still return a list here
        >>> select_subset(processors_to_run, "0") # to select only the first
        [1]

        >>> select_subset(processors_to_run, "-1") # to select only the last
        [5]

    Args:
        input_list (list): input list to select objects from.
        select_str (str): string representing Python slice.

    Returns:
        list: a subset of the input according to the ``select_str``

    """
    if ":" not in select_str:
        selected_objects = [input_list[int(select_str)]]
    else:
        slice_obj = slice(*map(lambda x: int(x.strip()) if x.strip() else None, select_str.split(":")))
        selected_objects = input_list[slice_obj]
    return selected_objects


def run_processors(cfg):
    logger.info(f"Hydra config: {OmegaConf.to_yaml(cfg)}")

    # Handle import manager if enabled
    if cfg.get("use_import_manager", False):
        try:
            import yaml
            yaml_path = cfg.get("config_path")
            if not yaml_path:
                raise ValueError("No configuration path provided in 'config_path'. Please specify the path.")

            if not os.path.exists(yaml_path):
                raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
            
            logger.info(f"Managing imports for config: {yaml_path}")
            manager = ImportManager()
            manager.sync_with_config(yaml_path)
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
        except ValueError as e:
            logger.error(f"Invalid configuration: {e}")
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file: {e}")
        except ImportError as e:
            logger.error(f"Import-related error: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during management of imports: {e}")

    # Detecting dask
    try:
        from dask.distributed import Client
        dask_available = True
    except ImportError:
        logger.warning("Dask not installed; using multiprocessing for all processors")
        dask_available = False
    
    # look for global directions in cfg for dask usage
    global_use_dask = bool(cfg.get("use_dask", True)) and dask_available

    processors_to_run = cfg.get("processors_to_run", "all")
    if processors_to_run == "all":
        processors_to_run = ":"
    selected_cfgs = select_subset(cfg.processors, processors_to_run)
    
    # filtering out any processors that have should_run=False
    processors_cfgs = []
    for processor_cfg in selected_cfgs:
        with open_dict(processor_cfg):
            should_run = processor_cfg.pop("should_run", True)
        if should_run:
            processors_cfgs.append(processor_cfg)

    logger.info(
        "Specified to run the following processors: %s ",
        [proc_cfg["_target_"] for proc_cfg in processors_cfgs],
    )
    
    
    
    processors = []
    # Create a temporary directory to hold intermediate files if needed.
    with tempfile.TemporaryDirectory() as tmp_dir:
        # special check for the first processor.
        # In case user selected something that does not start from
        # manifest creation we will try to infer the input from previous
        # output file
        if processors_cfgs[0] is not cfg.processors[0] and "input_manifest_file" not in processors_cfgs[0]:
            # locating starting processor
            for idx, processor in enumerate(cfg.processors):
                if processor is processors_cfgs[0]:  # we don't do a copy, so can just check object ids
                    if "output_manifest_file" in cfg.processors[idx - 1]:
                        with open_dict(processors_cfgs[0]):
                            processors_cfgs[0]["input_manifest_file"] = cfg.processors[idx - 1]["output_manifest_file"]
                    break
        
        for idx, processor_cfg in enumerate(processors_cfgs):
            logger.info('=> Building processor "%s"', processor_cfg["_target_"])

            # we assume that each processor defines "output_manifest_file"
            # and "input_manifest_file" keys, which can be optional. In case they
            # are missing, we create tmp files here for them
            # (1) first use a temporary file for the "output_manifest_file" if it is unspecified
            if "output_manifest_file" not in processor_cfg:
                tmp_file_path = os.path.join(tmp_dir, str(uuid.uuid4()))
                with open_dict(processor_cfg):
                    processor_cfg["output_manifest_file"] = tmp_file_path

            # (2) then link the current processor's output_manifest_file to the next processor's input_manifest_file
            # if it hasn't been specified (and if you are not on the last processor)
            if idx != len(processors_cfgs) - 1 and "input_manifest_file" not in processors_cfgs[idx + 1]:
                with open_dict(processors_cfgs[idx + 1]):
                    processors_cfgs[idx + 1]["input_manifest_file"] = processor_cfg["output_manifest_file"]
            
            #check if we have processor level directions of using dask
            flag=processor_cfg.get("use_dask", None)

            # if no processor-specific flag, fallback to global; otherwise use provided value
            if flag is None:
                use_dask_flag = global_use_dask
            else:
                use_dask_flag = flag

            processor = hydra.utils.instantiate(processor_cfg)
            processor.use_dask = use_dask_flag
            # running runtime tests to fail right-away if something is not
            # matching users expectations
            processor.test()
            processors.append(processor)


        # Start Dask client if any processor requires it
        dask_client = None
        if any(p.use_dask for p in processors):
            try:
                num_cpus = psutil.cpu_count(logical=False) or 4
                logger.info(f"Starting Dask client with {num_cpus} workers")
                dask_client = Client(n_workers=num_cpus, processes=True)
                logger.info(f"Dask dashboard at: {dask_client.dashboard_link}")
            except Exception as e:
                logger.warning(f"Failed to start Dask client: {e}")
                dask_client = None

        # Run processors in order
        try:
            for proc in processors:
                if proc.use_dask and dask_client is not None:
                    proc.dask_client = dask_client
                    logger.info('=> Running processor "%s" with Dask', proc)
                else:
                    logger.info('=> Running processor "%s" with Multiprocessing', proc)
                proc.process()
        finally:
            if dask_client is not None:
                logger.info("Shutting down Dask client...")
                dask_client.close(timeout="60s")
                logger.info("Dask client shutdown complete")

#tmp_dir is removed here after all processing finishes. !!!

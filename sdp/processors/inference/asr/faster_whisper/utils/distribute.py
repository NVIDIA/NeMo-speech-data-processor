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

import os
import socket
import torch
import torch.distributed as dist

from .logger import logger

def find_free_port(default_port=12355):
    """ Finds an available port for distributed communication. """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if s.connect_ex(("127.0.0.1", default_port)) != 0:
            return default_port  # Default port is available
        else:
            s.bind(("127.0.0.1", 0))  # Bind to a random available port
            return s.getsockname()[1]  # Return the assigned port

def validate_device_ids(device_ids):
    """ Validates that device_ids are within available GPU range. """
    available_gpus = torch.cuda.device_count()
    
    if not available_gpus:
        logger.error("No available CUDA devices found.")
        raise RuntimeError("No CUDA devices available.")

    # If device_ids is -1, select all GPUs
    if device_ids == -1:
        logger.info("Using all available GPUs.")
        return list(range(available_gpus))

    # Validate provided device IDs
    valid_devices = [d for d in device_ids if 0 <= d < available_gpus]

    if len(valid_devices) != len(device_ids):
        invalid_devices = set(device_ids) - set(valid_devices)
        logger.warning(f"Invalid GPU IDs detected: {invalid_devices}. Available GPUs: {list(range(available_gpus))}")
        
        if not valid_devices:
            raise ValueError("All specified GPU IDs are invalid. Check available devices.")

    return valid_devices

def init_distributed(device_ids, task_function, **task_kwargs):
    """ Initializes distributed processing and runs the specified task function on given devices. """
    if torch.cuda.is_available():
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(find_free_port())

        # Validate GPU IDs, handle -1 for all devices
        device_ids = validate_device_ids(device_ids)

        logger.info(f"Initializing distributed processing on devices: {device_ids}")

        results = torch.multiprocessing.spawn(
            run_distributed, 
            args=(device_ids, task_function, task_kwargs), 
            nprocs=len(device_ids), 
            join=True,
            return_result=True
        )
        return results
    else:
        logger.warning("CUDA is not available. Running in single-device mode.")
        return run_distributed(0, [0], task_function, task_kwargs)

def run_distributed(rank, device_ids, task_function, **task_kwargs):
    """ Runs distributed processing on each process with specific device IDs. """
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(len(device_ids))

    local_device = device_ids[rank]  # Assign specific GPU for this process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_device)

    dist.init_process_group("nccl", rank=rank, world_size=len(device_ids))
    torch.cuda.set_device(local_device)
    
    logger.info(f"Process {rank}/{len(device_ids)} initialized on GPU {local_device}.")

    # Call the provided task function
    result = task_function(**kwargs)
    results = [None] * len(device_ids)
    dist.all_gather_object(results, result)
    
    dist.destroy_process_group()
    logger.info(f"Process {rank} finalized.")
   
    if rank == 0:
        return results
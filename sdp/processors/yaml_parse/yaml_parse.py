import os
import yaml
import json
import tempfile
from typing import List, Optional
from braceexpand import braceexpand
from pathlib import Path

from sdp.logging import logger
from sdp.processors.base_processor import BaseProcessor
from sdp.run_processors import run_processors
from omegaconf import OmegaConf


class YamlParse(BaseProcessor):
    """Processor that runs a pipeline on each manifest file from a YAML file.
    
    This processor will:
    1. Parse the input YAML file to extract manifest filepaths
    2. For each manifest file, run the specified processors
    
    Args:
        input_yml (str): Path to the YAML file containing manifest filepath patterns
        processors_parseyaml: List of processor configurations to run on each manifest
        output_prefix (str, optional): Prefix to add to the output manifest paths
        max_files (int, optional): Maximum number of files to process per pattern

    Example:
        .. code-block:: yaml

            - _target_: sdp.processors.YamlParse
              input_yml: /pathto/test.yml 
              output_prefix: proc/ #can be folder or just prefix. In current example it will put output manifests on the same level but in folder proc
              processors_parseyaml:
                - _target_: sdp.processors.modify_manifest.common.DuplicateFields
                  output_manifest_file: None
                  duplicate_fields: 
                    id: Renamed

                - _target_: sdp.processors.GetAudioDuration
                  audio_filepath_key: audio_filepath
                  duration_key: duration
                  output_manifest_file: None
    """

    def __init__(
        self,
        input_yml: str,
        processors_parseyaml: list,
        output_prefix: str = "",
        max_files: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_yml = input_yml
        self.processors_parseyaml = processors_parseyaml
        self.output_prefix = output_prefix
        self.max_files = max_files

    def extract_manifest_files(self, source_pattern: str) -> List[str]:
        """Expand a pattern to a list of manifest files using braceexpand."""
        if "_OP_" in source_pattern:
            source_pattern = source_pattern.replace("_OP_", "{")
        if "_CL_" in source_pattern:
            source_pattern = source_pattern.replace("_CL_", "}")
        
        source_files = list(braceexpand(source_pattern))
        
        # Limit if max_files is set
        if self.max_files is not None and len(source_files) > self.max_files:
            source_files = source_files[:self.max_files]
            
        # Filter to make sure files exist
        existing_files = [f for f in source_files if os.path.exists(f)]
        if len(existing_files) < len(source_files):
            logger.warning(f"Some expanded manifest files don't exist. Found {len(existing_files)} of {len(source_files)}")
            
        return existing_files

    def read_yaml_file(self) -> List[dict]:
        """Parse the YAML file and extract all manifest filepaths and their config."""
        with open(self.input_yml, 'r') as f:
            cfg = yaml.safe_load(f)
            
        manifest_entries = []
        for input_cfg in cfg:
            for manifest in input_cfg.get('input_cfg', []):
                path = manifest.get('manifest_filepath')
                if path:
                    source_files = self.extract_manifest_files(path)
                    for src_file in source_files:
                        # Store both the file path and any additional config info
                        manifest_entries.append({
                            'path': src_file,
                            'config': manifest
                        })
        
        return manifest_entries

    def clean_config_params(self, config: dict) -> dict:
        """Clean configuration parameters, removing runtime attributes."""
        # List of internal or runtime attributes that shouldn't be passed to constructors
        exclude_params = {
            "max_workers", "chunksize", "in_memory_chunksize", 
            "number_of_entries", "total_duration", "test_cases",
            "start_time", "input_manifest_file", "output_manifest_file"
        }
        
        # Create a clean copy without these attributes
        return {k: v for k, v in config.items() if k not in exclude_params}
        #Better way to achieve this may exist #TODO

    def get_processor_config(self, processor):
        """Extract clean configuration from a processor object or dictionary."""
        if isinstance(processor, dict):
            return self.clean_config_params(processor)
        else:
            # For processor objects, extract class name and relevant configuration
            config = {}
            
            # Set target
            config["_target_"] = f"{processor.__class__.__module__}.{processor.__class__.__name__}"
            
            # Copy only relevant attributes from the processor object
            for key, value in processor.__dict__.items():
                # Skip internal attributes and those that shouldn't be part of configuration
                if not key.startswith('_') and key not in {
                    "max_workers", "chunksize", "in_memory_chunksize", 
                    "number_of_entries", "total_duration", "test_cases",
                    "start_time", "input_manifest_file", "output_manifest_file"
                }:
                    config[key] = value
                    
            return config

    def process(self):
        """Run processors on each manifest file from the YAML."""
        manifest_entries = self.read_yaml_file()
        logger.info(f"Found {len(manifest_entries)} manifest files to process")
        
        # Create output directory
        if self.output_manifest_file:
            os.makedirs(os.path.dirname(self.output_manifest_file), exist_ok=True)
            with open(self.output_manifest_file, 'w') as f:
                pass  # Create empty file for appending
        
        # Process each manifest file
        processed_count = 0
        for idx, entry in enumerate(manifest_entries):
            manifest_file = entry['path']
            manifest_config = entry['config']
            
            logger.info(f"Processing manifest file {idx+1}/{len(manifest_entries)}: {manifest_file}")
            
            
            manifest_dir = os.path.dirname(manifest_file)
            manifest_name = os.path.basename(manifest_file)
            
            # Create output path with prefix
            output_file = os.path.join(manifest_dir, f"{self.output_prefix}{manifest_name}")
            
            # Create processor configurations for this manifest
            processor_configs = []
            for i, proc in enumerate(self.processors_parseyaml):
                # Clean configuration
                config = self.get_processor_config(proc)
                
                # Set input manifest for first processor
                if i == 0:
                    config["input_manifest_file"] = manifest_file
                
                # Set output manifest for last processor
                if i == len(self.processors_parseyaml) - 1:
                    config["output_manifest_file"] = output_file
                
                processor_configs.append(config)
            
            # Create configuration for this run
            run_config = OmegaConf.create({
                "processors_to_run": "all",
                "processors": processor_configs
            })
            
            # Run processors
            try:
                run_processors(run_config)
                processed_count += 1
                logger.info(f"Successfully processed: {manifest_file} -> {output_file}")
                
                        
            except Exception as e:
                logger.error(f"Error processing {manifest_file}: {str(e)}")
                if self.output_manifest_file:
                    with open(self.output_manifest_file, 'a') as f:
                        f.write(json.dumps({
                            "original_manifest": manifest_file,
                            "status": "error",
                            "error": str(e)
                        }) + "\n")
        
        logger.info(f"Processed {processed_count}/{len(manifest_entries)} manifest files")


class CreateManifestListFromYaml(BaseProcessor):
    """Processor that parses a YAML file to extract manifest filepaths.
    
    This processor will:
    1. Parse the input YAML file
    2. Extract manifest filepaths using braceexpand for pattern expansion
    3. Save the list of manifest paths to the output file
    
    Args:
        input_yml (str): Path to the YAML file containing manifest filepath patterns
        max_files (int, optional): Maximum number of files to process per pattern. Defaults to None (all files).


    Example:
        .. code-block:: yaml

            - _target_: sdp.processors.CreateManifestListFromYaml
              input_yml: /test/test.yml
              output_manifest_file: /test/list.jsonl


    """

    def __init__(
        self,
        input_yml: str,
        max_files: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_yml = input_yml
        self.max_files = max_files

    def extract_manifest_files(self, source_pattern: str) -> List[str]:
        """Expand a pattern to a list of manifest files using braceexpand."""
        if "_OP_" in source_pattern:
            source_pattern = source_pattern.replace("_OP_", "{")
        if "_CL_" in source_pattern:
            source_pattern = source_pattern.replace("_CL_", "}")
        
        source_files = list(braceexpand(source_pattern))
        
        # Limit if max_files is set
        if self.max_files is not None and len(source_files) > self.max_files:
            source_files = source_files[:self.max_files]
            
        # Filter to make sure files exist
        existing_files = [f for f in source_files if os.path.exists(f)]
        if len(existing_files) < len(source_files):
            logger.warning(f"Some expanded manifest files don't exist. Found {len(existing_files)} of {len(source_files)}")
            
        return existing_files

    def read_yaml_file(self) -> List[str]:
        """Parse the YAML file and extract all manifest filepaths."""
        with open(self.input_yml, 'r') as f:
            cfg = yaml.safe_load(f)
            
        manifest_files = []
        for input_cfg in cfg:
            for manifest in input_cfg.get('input_cfg', []):
                path = manifest.get('manifest_filepath')
                if path:
                    source_files = self.extract_manifest_files(path)
                    manifest_files.extend(source_files)
        
        return manifest_files

    def process(self):
        """Extract manifest files from YAML and save the list to output."""
        manifest_files = self.read_yaml_file()
        logger.info(f"Found {len(manifest_files)} manifest files")
        
        # Ensure output directory exists
        if self.output_manifest_file:
            os.makedirs(os.path.dirname(self.output_manifest_file), exist_ok=True)
            
            # Write manifest paths to output file
            with open(self.output_manifest_file, 'w') as f:
                for manifest_file in manifest_files:
                    f.write(json.dumps({"manifest_path": manifest_file}) + "\n")
            
            logger.info(f"Saved list of {len(manifest_files)} manifest files to {self.output_manifest_file}")
        else:
            logger.warning("No output_manifest_file specified, manifest list not saved")
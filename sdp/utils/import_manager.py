import ast
import importlib
import inspect
import os
from pathlib import Path
from typing import Dict, Optional, Set
import yaml

from sdp.logging import logger

class ImportManager:
    """
    The ImportManager class is a utility designed to manage dynamic imports for a specific Python package based on a provided YAML configuration.
    This class simplifies the process of selectively importing only the necessary components,
    enabling the creation of a custom __init__.py file with imports for required processors. 
    By doing so, it ensures that users only need to install the libraries they actually use, 
    reducing unnecessary dependencies.
    
    To eable the ImportManager, set the `use_import_manager` key to `True` in the YAML config file. (Or provide it as an argument to main.py)
    use_import_manager: True

    """
    def __init__(self, base_package: str = "sdp"):
        self.base_package = base_package
        self.package_path = self._find_package_path()
        
    def _find_package_path(self) -> Path:
        try:
            package = importlib.import_module(self.base_package)
            return Path(package.__file__).parent
        except ImportError:
            current_dir = Path.cwd()
            for parent in [current_dir, *current_dir.parents]:
                if (parent / self.base_package).is_dir():
                    return parent / self.base_package
            raise FileNotFoundError(f"Could not find package '{self.base_package}'")

    def _get_processor_import(self, target: str) -> Optional[str]:
        try:
            module_path, class_name = target.rsplit('.', 1)
            return f"from {module_path} import {class_name}"
        except ValueError as e:
        # Raised if the target does not contain a '.'
            logger.warning(f"Invalid target format for import: '{target}'. Expected '<module>.<class>'. Error: {e}")
        except AttributeError as e:
        # Raised if the target module or class does not exist
            logger.warning(f"Invalid target type for import: {type(target)}. Error: {e}")
        except Exception as e:
            logger.warning(f"Could not process import for {target}: {e}")
        return None






    def get_required_imports(self, yaml_config: str) -> Set[str]:
        with open(yaml_config, 'r') as f:
            config = yaml.safe_load(f)
            
        required_imports = set()
        if 'processors' in config:
            for processor in config['processors']:
                if isinstance(processor, dict) and '_target_' in processor:
                    import_stmt = self._get_processor_import(processor['_target_'])
                    if import_stmt:
                        required_imports.add(import_stmt)
                        logger.debug(f"Found required processor: {processor['_target_']}")
        
        return required_imports

    def sync_with_config(self, yaml_config: str, init_file: Optional[str] = None) -> None:
        """
        Synchronize the __init__.py imports with the YAML config while preserving existing imports.
        """
        if init_file is None:
            init_file = self.package_path / 'processors' / '__init__.py'
        else:
            init_file = Path(init_file)

        logger.info(f"Syncing imports between {yaml_config} and {init_file}")

        # Get current content
        current_content = ""
        if init_file.exists():
            with open(init_file, 'r') as f:
                current_content = f.read()

        # Parse YAML config and get required imports
        required_imports = self.get_required_imports(yaml_config)
        
        # Mention that this file is auto-generated
        new_content = []
        if "let's import all supported processors" in current_content:
            # Keep the header comment if it exists
            new_content.append("# This was automaticly generated, to disable: set use_import_manager: False in yaml config\n")
        
        # Add imports
        for import_stmt in sorted(required_imports):
            new_content.append(import_stmt)
        
        # Write the new content
        init_file.parent.mkdir(parents=True, exist_ok=True)
        with open(init_file, 'w') as f:
            f.write('\n'.join(new_content))
        
        logger.info(f"Successfully updated {init_file} with required imports")


def setup_import_hooks():
    """Set up import hooks for automatic import management."""
    original_yaml_load = yaml.safe_load
    
    def yaml_load_hook(stream):
        result = original_yaml_load(stream)
        if isinstance(result, dict) and 'processors' in result:
            frame = inspect.currentframe()
            while frame:
                if frame.f_code.co_name != 'yaml_load_hook':
                    break
                frame = frame.f_back
            
            if frame:
                caller_file = frame.f_code.co_filename
                if isinstance(stream, str):
                    yaml_path = stream
                else:
                    yaml_path = os.path.abspath(caller_file)
                    
                manager = ImportManager()
                try:
                    manager.sync_with_config(yaml_path)
                except Exception as e:
                    logger.warning(f"Failed to sync imports: {e}")
        
        return result
    
    yaml.safe_load = yaml_load_hook
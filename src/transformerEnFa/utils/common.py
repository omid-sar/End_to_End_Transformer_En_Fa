import os
from box.exceptions import BoxValueError
import yaml
from transformerEnFa.logging import logger
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import re



@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")



@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"



class DirectoryTree:
    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.tree = []

    def _generate_tree(self, directory: Path, level: int = 0):
        """Helper recursive function to generate tree, excluding certain files/directories."""
        # Skip directories like '__pycache__', '.git', and other unwanted patterns
        if re.match(r'(__pycache__|\.git|\.github|.DS_Store)', directory.name):
            return

        indent = "  " * level
        self.tree.append(f"{indent}{directory.name}/\n")

        for item in directory.iterdir():
            if item.is_dir():
                self._generate_tree(item, level + 1)
            else:
                # Skip files with unwanted patterns
                if not re.match(r'(\.pyc|\.DS_Store|cache-|.*\.log|.*\.sample|FETCH_HEAD|ORIG_HEAD|COMMIT_EDITMSG)', item.name):
                    self.tree.append(f"{indent}  {item.name}\n")

    def write_to_file(self, file_output_name: str = "output.txt"):
        """Writes the tree structure to a file."""
        self._generate_tree(self.root_path)

        with open(file_output_name, 'w') as output_file:
            output_file.writelines(self.tree)





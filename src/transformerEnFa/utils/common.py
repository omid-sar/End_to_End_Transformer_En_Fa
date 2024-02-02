import os
from box.exceptions import BoxValueError
import yaml
from textSummarizer.logging import logger
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import re


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




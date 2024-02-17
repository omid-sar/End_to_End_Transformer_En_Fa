from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    dataset_name: str
    local_data_file: Path

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    ALL_REQUIRED_FILES: list 

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    tokenizer_file: Path
    local_data_file: Path
    lang_src: str
    lang_tgt: str
    seq_len: int
    batch_size: int
    train_val_split_ratio: Tuple[float, float]  

@dataclass(frozen=True)
class ModelConfig:
    root_dir: Path
    verification_info_dir: Path
    verification_summary_file: Path
    verification_weights_file: Path
    src_seq_len: int
    tgt_seq_len: int
    d_model: int
    N: int
    h: int
    dropout: float
    d_ff: int

@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: Path
    model_folder: Path
    model_basename: str
    tensorboard_log_dir: Path
    lr: float
    max_len: int
    preload: str
    num_epochs: int

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path

@dataclass(frozen=True)
class ModelInferenceConfig:
    root_dir: Path
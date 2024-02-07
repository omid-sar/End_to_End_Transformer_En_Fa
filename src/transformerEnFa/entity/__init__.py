from dataclasses import dataclass
from pathlib import Path


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
class ModelConfig:
    root_dir: Path
    verification_info_dir: Path
    verification_summary_file: Path
    verification_weights_file: Path
    src_vocab_size: int
    tgt_vocab_size: int
    src_seq_len: int
    tgt_seq_len: int
    d_model: int
    N: int
    h: int
    dropout: float
    d_ff: int
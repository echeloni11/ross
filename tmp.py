import os
import io
import copy
import sys
from dataclasses import dataclass, field
from collections import OrderedDict
import datetime
import builtins
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import torch
import torch.distributed as dist

import transformers
import tokenizers
import megfile

from ross.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from torch.utils.data import Dataset
from ross.ross_trainer import RossTrainer

from ross import conversation as conversation_lib
from ross.model import *
from ross.mm_utils import tokenizer_image_token

parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))

model_args, data_args, training_args = parser.parse_args_into_dataclasses()

for arg in [model_args, data_args, training_args]:
    print(arg)
# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
SFT dataset
- We assume user pass a single parquet file.
- We load all the data into the memory.
Each parquet file contains
"""

from typing import Union

import pandas as pd
import torch
from omegaconf.listconfig import ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask


class SFTDataset(Dataset):
    """
    This is an in-memory SFTDataset

    Arguments:
        config (OmegaConf): the data config
    """

    def __init__(self, parquet_files: Union[str, ListConfig], tokenizer, config):
        prompt_key = config.get("prompt_key", "prompt")
        prompt_dict_keys = config.get("prompt_dict_keys", None)
        response_key = config.get("response_key", "response")
        response_dict_keys = config.get("response_dict_keys", None)
        max_length = config.get("max_length", 1024)
        truncation = config.get("truncation", "error")
        use_shm = config.get("use_shm", False)

        assert truncation in ["error", "left", "right"]
        self.truncation = truncation
        self.use_shm = use_shm

        if not isinstance(parquet_files, ListConfig):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self.prompt_key = prompt_key if isinstance(prompt_key, (tuple, list)) else [prompt_key]
        self.response_key = response_key if isinstance(response_key, (tuple, list)) else [response_key]
        self.prompt_dict_keys = prompt_dict_keys if prompt_dict_keys else []
        self.response_dict_keys = response_dict_keys if response_dict_keys else []

        self.max_length = max_length

        self._download()
        self._read_files_and_tokenize()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_to_local(parquet_file, verbose=True, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        def series_to_item(ls):
            import numpy
            import pandas

            while isinstance(ls, (pandas.core.series.Series, numpy.ndarray)) and len(ls) == 1:
                ls = ls[0]
            return ls

        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)
        self.prompts = self.dataframe[self.prompt_key]
        for key in self.prompt_dict_keys:
            # type(x): pandas.core.series.Series
            # type(x[0]): numpy.ndarray
            # type(x[0][0]): dict
            try:
                self.prompts = self.prompts.apply(lambda x: series_to_item(x)[key], axis=1)  # noqa: B023
            except Exception:
                print(f"self.prompts={self.prompts}")
                raise
        if isinstance(self.prompts, pd.DataFrame):
            self.prompts = self.prompts.squeeze()
        self.prompts = self.prompts.tolist()
        self.responses = self.dataframe[self.response_key]
        for key in self.response_dict_keys:
            try:
                self.responses = self.responses.apply(lambda x: series_to_item(x)[key], axis=1)  # noqa: B023
            except Exception:
                print(f"self.responses={self.responses}")
                raise
        if isinstance(self.responses, pd.DataFrame):
            self.responses = self.responses.squeeze()
        self.responses = self.responses.tolist()

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, item):
        tokenizer = self.tokenizer

        prompt = self.prompts[item]
        response = self.responses[item]

        # apply chat template
        prompt_chat = [{"role": "user", "content": prompt}]

        # string
        prompt_chat_str = tokenizer.apply_chat_template(prompt_chat, add_generation_prompt=True, tokenize=False)
        response_chat_str = response + tokenizer.eos_token

        # tokenize
        prompt_ids_output = tokenizer(prompt_chat_str, return_tensors="pt", add_special_tokens=False)
        prompt_ids = prompt_ids_output["input_ids"][0]
        prompt_attention_mask = prompt_ids_output["attention_mask"][0]

        response_ids_output = tokenizer(response_chat_str, return_tensors="pt", add_special_tokens=False)
        response_ids = response_ids_output["input_ids"][0]
        response_attention_mask = response_ids_output["attention_mask"][0]

        prompt_length = prompt_ids.shape[0]
        response_length = response_ids.shape[0]

        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)

        # padding to max length
        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            padded_input_ids = (
                torch.ones(size=(self.max_length - sequence_length,), dtype=input_ids.dtype)
                * self.tokenizer.pad_token_id
            )
            padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)

            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
        elif sequence_length > self.max_length:
            if self.truncation == "left":
                # actually, left truncation may not be reasonable
                input_ids = input_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
            elif self.truncation == "error":
                raise NotImplementedError(f"{sequence_length=} is larger than {self.max_length=}")
            else:
                raise NotImplementedError(f"Unknown truncation method {self.truncation}")

        position_ids = compute_position_id_with_mask(attention_mask)

        loss_mask = attention_mask.clone()
        if prompt_length > 1:
            # mask out prompt for SFT.
            loss_mask[: min(prompt_length, loss_mask.size(0)) - 1] = 0
        # mask out the last token in response
        loss_mask[min(prompt_length + response_length, loss_mask.size(0)) - 1] = 0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }


class NegWeightSFTDataset(SFTDataset):
    """
    SFTDataset variant that supports negative sample weighting.

    For each sample, it reads a ``label`` field (either from the top-level column
    or from ``extra_info['label']``) to decide whether the sample is correct/positive
    or incorrect/negative.

    Positive samples  → sample_weight =  1.0  (normal SFT loss)
    Negative samples  → sample_weight = -neg_weight  (e.g. -0.2, to *minimise* their log-prob)

    The ``sample_weight`` scalar tensor is returned in each data dict so that the
    trainer can multiply it into the per-sample loss before aggregation.

    Extra config keys (under ``data``):
        label_key (str):         column name that stores the label, default ``"label"``
        label_dict_keys (list):  if label is nested in a dict column, list of keys to
                                 traverse (similar to prompt_dict_keys), default ``[]``
        positive_label (str):    string value that marks a positive sample, default ``"correct"``
        neg_weight (float):      absolute weight applied to negative samples (the actual
                                 multiplier will be ``-neg_weight``), default ``0.2``
    """

    def __init__(self, parquet_files: Union[str, ListConfig], tokenizer, config):
        # Read neg-weight specific config before calling super().__init__ so that
        # the extra columns are available after _read_files_and_tokenize().
        self._label_key = config.get("label_key", "label")
        self._label_dict_keys = list(config.get("label_dict_keys", []))
        self._positive_label = config.get("positive_label", "correct")
        self._neg_weight = float(config.get("neg_weight", 0.2))

        super().__init__(parquet_files=parquet_files, tokenizer=tokenizer, config=config)

    def _read_files_and_tokenize(self):
        # Let the parent class load prompts / responses / dataframe first.
        super()._read_files_and_tokenize()

        def series_to_item(ls):
            import numpy
            import pandas

            while isinstance(ls, (pandas.core.series.Series, numpy.ndarray)) and len(ls) == 1:
                ls = ls[0]
            return ls

        # Extract label column
        labels_col = self.dataframe[self._label_key]
        for key in self._label_dict_keys:
            labels_col = labels_col.apply(lambda x: series_to_item(x)[key])  # noqa: B023

        self.labels = labels_col.tolist()

    def __getitem__(self, item):
        data = super().__getitem__(item)

        label = self.labels[item]
        # Positive sample → weight 1.0; negative sample → weight -neg_weight
        if label == self._positive_label:
            weight = 1.0
        else:
            weight = -self._neg_weight

        data["sample_weight"] = torch.tensor(weight, dtype=torch.float32)
        return data


class WeightedSFTDataset(SFTDataset):
    """
    SFTDataset variant that reads a per-sample weight directly from a column in the
    parquet file (e.g. ``weight`` or ``weight_norm``).

    Unlike ``NegWeightSFTDataset`` which derives the sign from a binary label column,
    this dataset expects the weight to already carry its sign:
      - Positive weight  → treat as a "correct" sample; loss = weight * (-p)
      - Negative weight  → treat as an "incorrect" sample; loss = |weight| * (-p)
      (both branches use the same formula; the sign is handled in the trainer)

    Extra config keys (under ``data``):
        weight_key (str):        column name storing the scalar weight, default ``"weight"``
        weight_dict_keys (list): if weight is nested in a dict column, list of keys to
                                 traverse (similar to prompt_dict_keys), default ``[]``
    """

    def __init__(self, parquet_files: Union[str, ListConfig], tokenizer, config):
        self._weight_key = config.get("weight_key", "weight")
        self._weight_dict_keys = list(config.get("weight_dict_keys", []))

        super().__init__(parquet_files=parquet_files, tokenizer=tokenizer, config=config)

    def _read_files_and_tokenize(self):
        # Let the parent class load prompts / responses / dataframe first.
        super()._read_files_and_tokenize()

        def series_to_item(ls):
            import numpy
            import pandas

            while isinstance(ls, (pandas.core.series.Series, numpy.ndarray)) and len(ls) == 1:
                ls = ls[0]
            return ls

        # Extract weight column
        weights_col = self.dataframe[self._weight_key]
        for key in self._weight_dict_keys:
            weights_col = weights_col.apply(lambda x: series_to_item(x)[key])  # noqa: B023

        self.sample_weights = weights_col.tolist()

    def __getitem__(self, item):
        data = super().__getitem__(item)
        weight = float(self.sample_weights[item])
        data["sample_weight"] = torch.tensor(weight, dtype=torch.float32)
        return data

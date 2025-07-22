import glob
import pathlib
import shutil
from collections.abc import KeysView
from typing import Any, Dict, List, Optional, Union

import torch
from termcolor import cprint


class TensorCache:
    def __init__(self, keys: List[str]) -> None:
        self._storage: Dict[str, List[torch.Tensor]] = {}
        self._len = 0
        self._keys = keys
        self.reset()

    def reset(self) -> None:
        self._storage = {k: [] for k in self._keys}
        self._len = 0

    def _check_valid_tensor(self, tensor: torch.Tensor, k: str) -> None:
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(
                f"Invalid data for key {k}. "
                "Metrics cache only accepts torch tensor data."
            )
        stored_for_k = self._storage[k]
        if len(stored_for_k) > 0 and stored_for_k[-1].shape != tensor.shape:
            raise ValueError(
                f"Invalid data for key {k}. "
                f"Expected tensor of shape {stored_for_k[-1]} but got {tensor.shape}."
            )

    def append(self, data: Dict[str, torch.Tensor]) -> None:
        if not self._storage.keys() == data.keys():
            raise ValueError(
                "MetricsCache only allows appending data with the same keys "
                "as used at construction time."
            )
        for k in data.keys():
            tensor = data[k]
            self._check_valid_tensor(tensor, k)
            self._storage[k].append(tensor.detach().clone().cpu())
        self._len += 1

    def __getitem__(
        self, key: Union[int, str]
    ) -> Union[List[torch.Tensor], Dict[str, torch.Tensor]]:
        if isinstance(key, str):
            return self._storage[key]
        return {k: v[key] for k, v in self._storage.items()}

    def __len__(self) -> int:
        return self._len

    def dump(self, fname: str) -> None:
        torch.save({k: torch.stack(v, dim=-1) for k, v in self._storage.items()}, fname)

    def size(self) -> int:
        sz = 0
        for _, tensor_list in self._storage.items():
            if len(tensor_list) == 0:
                continue
            t = tensor_list[0]
            sz += t.element_size() * t.nelement()
        return sz * len(self)

    def keys(self) -> KeysView:
        return self._storage.keys()


class TensorLogger:
    _CONSTANTS_FILENAME = "constants.pth"

    def __init__(
        self,
        save_dir: Optional[Union[str, pathlib.Path]],
        keys: Optional[List[str]] = None,
        max_capacity: int = 10000000,  # 10MBs
    ) -> None:
        """Logs torch tensor histories that will be mapped to the specified keys.

        Args:
            keys (List[str], optional): the name of the attributes to log.
                If not provided, the keys will be taken from the first call to update
                and then locked (subsequent calls must respect abide by these keys).
            save_dir (Optional[Union[str, pathlib.Path]]): where to save the logs.
            max_capacity (int, optional): storage will be dumped to disk once they
                are over this size in bytes. Defaults to 10MB.

        Raises:
            ValueError: if any given key is not an attribute of the target object.
        """
        self.active = True
        if save_dir is None:
            self.active = False
            cprint("No metrics log dir given, logger will be inactive.", "yellow")
            return
        else:
            self.save_path = pathlib.Path(save_dir)
            if self.save_path.exists():
                if not self.save_path.is_dir():
                    raise ValueError("Save path exists and it is not a directory.")
            cprint(f"Saving metrics log to {self.save_path}.", "green")
            if self.save_path.exists():
                cprint(f"Deleting previous content.", "yellow")
                shutil.rmtree(self.save_path)
        self.save_path.mkdir(exist_ok=True, parents=True)
        self.keys = keys
        self.cache: TensorCache = None
        if self.keys is not None:
            self.cache = TensorCache(keys)
        # This is used to hold the tensors to update the logger before syncing
        self._tmp_tensor_dict: Dict[str, torch.Tensor] = {}
        self.max_capacity = max_capacity
        self._cur_cache_idx = 0
        self._constants_dict: Dict[str, torch.Tensor] = {}

    @staticmethod
    def _fname_from_idx(idx: int) -> str:
        return f"{idx:04d}.pth"

    def _dump_path(self) -> str:
        return self.save_path / TensorLogger._fname_from_idx(self._cur_cache_idx)

    def add_constant(self, name: str, data: Any, strict: bool = False) -> None:
        if name in self._constants_dict:
            if strict:
                raise ValueError(
                    f"A constant with name {name} is already in the logger."
                )
            else:
                return
        try:
            self._constants_dict[name] = torch.as_tensor(data)
        except RuntimeError:
            self._constants_dict[name] = data

    def update(self, tensor_dict: Dict[str, torch.Tensor], sync: bool = True) -> None:
        """Updates the tensor cache with the given entries.

        The cache is only updated if sync is True (the default). Otherwise, a temp
        dict accumulates tensors until a call to update(sync=True) happens.
        """
        if not self.active:
            return

        self._tmp_tensor_dict.update(tensor_dict)
        if not sync:
            return
        if self.cache is None:
            self.cache = TensorCache(self._tmp_tensor_dict.keys())
        if not self._tmp_tensor_dict.keys() == self.cache.keys():
            raise ValueError(
                f"The input dict's keys dict don't match the expected ones {self.keys}"
            )
        self.cache.append(self._tmp_tensor_dict)
        if self.cache.size() > self.max_capacity:
            self.cache.dump(self._dump_path())
            self.cache.reset()
            self._cur_cache_idx += 1
        self._tmp_tensor_dict = {}  # empty the temporary tensor dict until next sync

    def close(self) -> None:
        if not self.active:
            return
        if self.cache is not None and len(self.cache) > 0:
            self.cache.dump(self._dump_path())
        if self._constants_dict:
            torch.save(
                self._constants_dict, self.save_path / TensorLogger._CONSTANTS_FILENAME
            )

    @staticmethod
    def load(
        load_path: Union[str, pathlib.Path], lazy: bool = False
    ) -> Dict[str, torch.Tensor]:
        if lazy:
            raise NotImplementedError("Lazy cache file access is not implemented.")
        load_path = pathlib.Path(load_path)
        cache_files = glob.glob(f"{load_path}/[0-9]*.pth")
        results: Dict[str, torch.Tensor] = None
        for fpath in cache_files:
            data = torch.load(fpath)
            if results is None:
                results = data
            else:
                assert (
                    results.keys() == data.keys()
                ), f"f{fpath}: Expected {results.keys()} but got {data.keys()}"
                results = {k: torch.cat([results[k], data[k]], -1) for k in results}
        constants = torch.load(load_path / TensorLogger._CONSTANTS_FILENAME)
        results.update(constants)
        return results

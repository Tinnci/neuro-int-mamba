from typing import Any
import numpy as np

class MjModel:
    @staticmethod
    def from_xml_path(path: str) -> 'MjModel': ...
    opt: Any
    timestep: float
    qpos: np.ndarray
    qvel: np.ndarray

class MjData:
    def __init__(self, model: MjModel) -> None: ...
    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    sensordata: np.ndarray

def mj_step(m: MjModel, d: MjData) -> None: ...

class _ViewerHandle:
    def is_running(self) -> bool: ...
    def sync(self) -> None: ...
    def __enter__(self) -> '_ViewerHandle': ...
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None: ...

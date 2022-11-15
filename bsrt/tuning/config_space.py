from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Dict, Union

import syne_tune.config_space as st

ConfigType = Union[st.Categorical, st.Float, st.Integer]


@dataclass
class ConfigSpace(ABC):
    def to_dict(self) -> Dict[str, ConfigType]:
        """
        Convert the config space to a dictionary of config types.
        Additionally, the keys are prefixed with the class name of the corresponding Params class.
        """
        prefix = self.__class__.__name__.replace("ConfigSpace", "Params.", 1)
        return {prefix + k: v for k, v in self.__dict__.items()}

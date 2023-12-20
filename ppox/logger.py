# Copyright 2022 InstaDeep Ltd. All rights reserved.
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

"""Logger setup."""

from typing import Dict, Optional, Protocol
import jax.numpy as jnp
import numpy as np
from colorama import Fore, Style
from datetime import datetime
import json
import logging
import os
import time

from ppox.types import ExperimentOutput

class JsonLogger:
    def __init__(self, cfg: Dict) -> None:
        self.should_log = cfg["logger"]["should_log"]
        self.console_logger = get_python_logger()

        if cfg["logger"]["json_path"] is not None:
            self.path = cfg["logger"]["base_exp_path"] + "/" + cfg["logger"]["json_path"]
        else:
            self.path = (
                f"{cfg['logger']['base_exp_path']}/"
                + f"{cfg['logger']['system_name']}/{cfg['env_name']}"
                + f"/envs_{cfg['num_envs']}/seed_{cfg['seed']}/"
                + f"{datetime.now().strftime('%Y%m%d%H%M%S')}"
            )

        self.file_name = "metrics.json"
        self.run_data: Dict = {"absolute_metrics": {}}

        # If the file already exists, load it
        if os.path.isfile(f"{self.path}/{self.file_name}"):
            with open(f"{self.path}/{self.file_name}", "r") as f:
                data = json.load(f)

        else:
            # Create the logging directory if it doesn't exist
            os.makedirs(self.path, exist_ok=True)
            data = {}

        algorithm_name = cfg["logger"]["system_name"]
        environment_name = cfg["env_name"]
        # Merge the existing data with the new data
        self.data = data
        if environment_name not in self.data:
            self.data[environment_name] = {}
        if algorithm_name not in self.data[environment_name]:
            self.data[environment_name][algorithm_name] = {}
        self.data[environment_name][algorithm_name][f"seed_{cfg['seed']}"] = self.run_data

        with open(f"{self.path}/{self.file_name}", "w") as f:
            json.dump(self.data, f, indent=4)


    def log_stat(
        self,
        key: str,
        value: float,
        timestep: int,
        evaluation_step: Optional[int] = None,
    ) -> None:
        """
        Writes a step to the json reporting file

        Args:
            key (str): the metric that should be logged
            value (str): the value of the metric that should be logged
            timestep (int): the current environment timestep
            evaluation_step (int): the evaluation step
        """

        current_time = time.time()

        # This will ensure the first logged time is 0, which avoids taking compilation into account
        # when plotting downstream.
        if evaluation_step == 0:
            self.start_time = current_time

        logging_prefix, *metric_key = key.split("/")
        metric_key = "/".join(metric_key)

        metrics = {metric_key: [value]}

        if logging_prefix == "evaluator":
            step_metrics = {"step_count": timestep, "elapsed_time": current_time - self.start_time}
            step_metrics.update(metrics)  # type: ignore
            step_str = f"step_{evaluation_step}"
            if step_str in self.run_data:
                self.run_data[step_str].update(step_metrics)
            else:
                self.run_data[step_str] = step_metrics

        # Store the absolute metrics
        if logging_prefix == "absolute":
            self.run_data["absolute_metrics"].update(metrics)

        with open(f"{self.path}/{self.file_name}", "w") as f:
            json.dump(self.data, f, indent=4)

def get_python_logger() -> logging.Logger:
    """Set up a custom python logger."""
    logger = logging.getLogger()
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter(f"{Fore.CYAN}{Style.BRIGHT}%(message)s", "%H:%M:%S")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # Set to info to suppress debug outputs.
    logger.setLevel("INFO")

    return logger


# Not in types.py because we only use it here.
class LogFn(Protocol):
    def __call__(
        self,
        metrics: ExperimentOutput,
        t_env: int = 0,
        trainer_metric: bool = False,
        absolute_metric: bool = False,
        eval_step: Optional[int] = None,
    ) -> float:
        ...


def get_logger_tools(logger: JsonLogger) -> LogFn:  
    """Get the logger function."""

    def log(
        metrics: ExperimentOutput,
        t_env: int = 0,
        trainer_metric: bool = False,
        absolute_metric: bool = False,
        eval_step: Optional[int] = None,
    ) -> float:
        """Log the episode returns and lengths.

        Args:
            metrics (Dict): The metrics info.
            t_env (int): The current environment timestep.
            trainer_metric (bool): Whether to log the trainer metric.
            absolute_metric (bool): Whether to log the absolute metric.
            eval_step (int): The count of the current evaluation.
        """
        if absolute_metric:
            prefix = "absolute/"
            episodes_info = metrics.episodes_info
        elif trainer_metric:
            prefix = "trainer/"
            episodes_info = metrics.episodes_info
            total_loss = metrics.total_loss
            value_loss = metrics.value_loss
            loss_actor = metrics.loss_actor
            entropy = metrics.entropy
        else:
            prefix = "evaluator/"
            episodes_info = metrics.episodes_info

        # Flatten metrics info.
        episodes_return = jnp.ravel(episodes_info["episode_return"])
        episodes_length = jnp.ravel(episodes_info["episode_length"])
        steps_per_second = episodes_info["steps_per_second"]

        # Log metrics.
        if logger.should_log:
            logger.log_stat(
                f"{prefix}mean_episode_returns", float(np.mean(episodes_return)), t_env, eval_step
            )
            logger.log_stat(
                f"{prefix}mean_episode_length", float(np.mean(episodes_length)), t_env, eval_step
            )
            logger.log_stat(f"{prefix}steps_per_second", steps_per_second, t_env, eval_step)

            if trainer_metric:
                logger.log_stat(f"{prefix}total_loss", float(np.mean(total_loss)), t_env)
                logger.log_stat(f"{prefix}value_loss", float(np.mean(value_loss)), t_env)
                logger.log_stat(f"{prefix}loss_actor", float(np.mean(loss_actor)), t_env)
                logger.log_stat(f"{prefix}entropy", float(np.mean(entropy)), t_env)

        log_string = (
            f"Timesteps {t_env:07d} | "
            f"Mean Episode Return {float(np.mean(episodes_return)):.3f} | "
            f"Std Episode Return {float(np.std(episodes_return)):.3f} | "
            f"Max Episode Return {float(np.max(episodes_return)):.3f} | "
            f"Steps Per Second {steps_per_second:.2e} "
        )
        # log_string = (
        #     f"Timesteps {t_env:07d} | "
        #     f"Mean Episode Return {float(np.mean(episodes_return)):.3f} | "
        #     f"Std Episode Return {float(np.std(episodes_return)):.3f} | "
        #     f"Max Episode Return {float(np.max(episodes_return)):.3f} | "
        #     f"Mean Episode Length {float(np.mean(episodes_length)):.3f} | "
        #     f"Std Episode Length {float(np.std(episodes_length)):.3f} | "
        #     f"Max Episode Length {float(np.max(episodes_length)):.3f} | "
        #     f"Steps Per Second {steps_per_second:.2e} "
        # )

        if absolute_metric:
            logger.console_logger.info(
                f"{Fore.BLUE}{Style.BRIGHT}ABSOLUTE METRIC: {log_string}{Style.RESET_ALL}"
            )
        elif trainer_metric:
            log_string += (
                f"| Total Loss {float(np.mean(total_loss)):.3f} | "
                f"Value Loss {float(np.mean(value_loss)):.3f} | "
                f"Loss Actor {float(np.mean(loss_actor)):.3f} | "
                f"Entropy {float(np.mean(entropy)):.3f}"
            )
            logger.console_logger.info(
                f"{Fore.MAGENTA}{Style.BRIGHT}TRAINER: {log_string}{Style.RESET_ALL}"
            )
        else:
            logger.console_logger.info(
                f"{Fore.GREEN}{Style.BRIGHT}EVALUATOR: {log_string}{Style.RESET_ALL}"
            )

        return float(np.mean(episodes_return))

    return log


def logger_setup(config: Dict) -> LogFn:
    """Setup the logger."""
    logger = JsonLogger(config)
    return get_logger_tools(logger)

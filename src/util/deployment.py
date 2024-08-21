import torch
import os
from warnings import warn
from datetime import datetime
import subprocess
from typing import Union, Optional
from src.models.base import BaseModel
from src.models.DimeNetPP import DimeNetPP
from src.models.SchNet import SchNet
from src.models.MACE import MACE


def remove_first_row(string):
    lines = string.split("\n")
    return "\n".join(lines[1:])


def save_model(model: BaseModel, file_path: str):
    save_dict = {}
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("utf-8")
            .strip()
        )
        save_dict["git_commit"] = commit
    except:
        warn("Could not get git commit hash. Will not store the commit hash then.")

    try:
        env = subprocess.check_output(["micromamba", "list"]).decode("utf-8").strip()
        env = remove_first_row(env)
        save_dict["env_info"] = env
    except:
        warn(
            "Could not get information on the virtual environment. "
            "Probably because micromamba is not installed."
            " Will not store information on the environment then."
        )

    # save time when model was exported
    save_dict["time_created"] = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

    # Save hyperparameters
    save_dict["hyperparameters"] = model.hyperparameters
    save_dict["state_dict"] = model.state_dict()
    torch.save(save_dict, file_path)


def restore_model(
    file_path: str,
    module_cls: Optional[type[BaseModel]] = None,
    strict: bool = True,
) -> Union[DimeNetPP]:
    """
    :param file_path: Path pointing to model file
    :param module_cls: Optional, class for loading the specific model.
    If none is provided, the class will be inferred.
    :param strict: strict-argument of load_state_dict
    :return: model instance
    """
    # Load the saved dictionary
    save_dict = torch.load(file_path)
    # Extract the hyperparameters and state dictionary
    hyperparameters = save_dict["hyperparameters"]
    state_dict = save_dict["state_dict"]

    if module_cls is not None:
        _module_cls = module_cls
    else:
        _module_cls = model_resolver(filepath=file_path)

    # Create an instance of the model with the loaded hyperparameters
    model = _module_cls(**hyperparameters)

    # Load the state dictionary into the model
    model.load_state_dict(state_dict, strict=strict)
    return model


def model_resolver(filepath):
    """
    Helper function for inferring the correct model class by the file extension.
    :param filepath: Path pointing to a model checkpoint
    :return: Model class
    """
    name, extension = os.path.splitext(filepath)
    mapping_dict = {
        "pt": DimeNetPP,  # for backwards compatibility
        "dimenet": DimeNetPP,
        "mace": MACE,
        "schnet": SchNet,
    }
    extension = extension[1:]
    assert extension in mapping_dict.keys(), (
        f"Could not find suitable model for file extension .{extension}. "
        f"Make sure that the file extension matches one in {mapping_dict.keys()}!"
    )
    if extension == "pt":
        print(
            f"Model file has extension .pt. Assume it is an instance of DimeNetSingle. "
            f"If this is not correct, change the file extension"
        )
    return mapping_dict[extension]

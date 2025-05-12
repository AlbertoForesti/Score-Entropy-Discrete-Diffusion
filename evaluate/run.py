import hydra
import mutinfo
import numpy

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from tqdm import trange

from infosedd.infosedd import InfoSEDD

import yaml

import torch
import bebeziana


def ndarray_representer(dumper: yaml.Dumper, array: numpy.ndarray) -> yaml.Node:
    return dumper.represent_list(array.tolist())

def register_values(values):
    try:
        values = numpy.array(values)
        mean = float(numpy.mean(values))
        std  = float(numpy.std(values))
        return values, mean, std
    except:
        values_dict = {}
        for value in values:
            if isinstance(value, dict):
                for key, value in value.items():
                    if key not in values_dict:
                        values_dict[key] = []
                    values_dict[key].append(float(value))
            else:
                raise ValueError("Expected an iterable of dictionaries or an iterable of values as output of estimator")
        return values_dict, {key: float(numpy.mean(values)) for key, values in values_dict.items()}, {key: float(numpy.std(values)) for key, values in values_dict.items()}
        


@hydra.main(version_base=None, config_path="./config.d", config_name="config")
def run_test(config : DictConfig) -> None:
    try:
        bebeziana.seed_everything(config["seed"], to_be_seeded=config["to_be_seeded"])

        # Resolving some parts of the config and storing them separately for later post-processing.
        setup = {}
        setup["estimator"]    = OmegaConf.to_container(config["estimator"], resolve=True)
        setup["distribution"] = OmegaConf.to_container(config["distribution"], resolve=True)

        # Results for post-processing.
        results = {}
        results["mutual_information"] = {"values": []}
        alphabet_size = config["alphabet_size"]
        if hasattr(config["estimator"], "args"):
            OmegaConf.update(config, "estimator.args.alphabet_size", alphabet_size, force_add=True)

        for index in trange(config["n_runs"]):
            random_variable = instantiate(config["distribution"])
            estimator       = instantiate(config["estimator"])

            x, y = random_variable.rvs(config["n_samples"])
            
            x = x.reshape(config["n_samples"], -1)
            y = y.reshape(config["n_samples"], -1)
            
            results["mutual_information"]["values"].append(estimator(x, y))
            torch.cuda.empty_cache()

        values, mean, std = register_values(results["mutual_information"]["values"])

        results["mutual_information"]["values"] = values
        results["mutual_information"]["mean"]   = mean
        if isinstance(estimator, InfoSEDD):
            results["mutual_information"]["std"]    = estimator.mutinfo_estimate_std
        else:
            results["mutual_information"]["std"]    = std

        path = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        with open(path / "setup.yaml", 'w') as file:
            yaml.dump(setup, file, default_flow_style=False)
        with open(path / "results.yaml", 'w') as file:
            yaml.dump(results, file, default_flow_style=False)
        print("Experiment completed successfully.")
    except Exception as exception:
        raise exception
        """
        print(exception) # Just ignoring exceptions so the sweeping continues."""



if __name__ == "__main__":
    yaml.add_representer(numpy.ndarray, ndarray_representer)
    run_test()
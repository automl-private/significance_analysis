from omegaconf import DictConfig
import hydra
from experimentPlatform import runExperiment

@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    #print(OmegaConf.to_yaml(cfg))
    #print(cfg["aquisitionFunction"]["aquisitionFunction"])
    configDict={
        "sobolRounds":cfg["experimentConfig"]["sobolRounds"],
        "botorchRounds":cfg["trainingRounds"]["botorchRounds"],
        "benchmark":cfg["benchmark"]["benchmark"],
        "randomSeed":"0",
        "aquisitionFunction":cfg["aquisitionFunction"]["aquisitionFunction"],
        "surrogateFunction":cfg["surrogateFunction"]["surrogateFunction"],
        "saveResults":cfg["experimentConfig"]["saveResults"]
    }
    runExperiment(configDict)
    return 0

if __name__ == "__main__":
    my_app()
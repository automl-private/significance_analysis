from omegaconf import DictConfig, OmegaConf
import hydra
from experimentPlatform import runExperiment

@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    configDict={
        "sobolRounds":cfg["experimentConfig"]["sobolRounds"], 
        "botorchRounds":cfg["experimentConfig"]["botorchRounds"],
        "benchmark":cfg["benchmark"]["benchmark"],
        "randomSeed":"0",
        "aquisitionFunction":cfg["aquisitionFunction"]["aquisitionFunction"],
        "surrogateFunction":cfg["surrogateFunction"]["surrogateFunction"],
        "executeTraining":cfg["experimentConfig"]["executeTraining"],
        "saveResults":cfg["experimentConfig"]["saveResults"]
    }
    print(configDict)
    runExperiment(configDict)

if __name__ == "__main__":
    my_app()
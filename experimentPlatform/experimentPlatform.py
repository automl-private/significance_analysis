#from hpobench.benchmarks.ml.nn_benchmark import NNBenchmark
import os
import numpy
import jahs_bench
import pandas as pd
from ax import (
    ChoiceParameter,
    Data,
    Experiment,
    FixedParameter,
    Metric,
    Objective,
    OptimizationConfig,
    ParameterType,
    RangeParameter,
    Runner,
    SearchSpace,
)
from ax.metrics.branin import BraninMetric
from ax.metrics.hartmann6 import Hartmann6Metric
from ax.modelbridge.registry import Models
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.utils.common.result import Ok
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
    qNoisyExpectedImprovement,
)
from botorch.models.gp_regression import FixedNoiseGP, SingleTaskGP





def runExperiment(hydraConfig:dict):
    #Unpacking dict (from Hydra)
    numberOfSobolRounds = int(hydraConfig["sobolRounds"])
    numberOfBotorchRounds = int(hydraConfig["botorchRounds"])
    chosenBenchmark = hydraConfig["benchmark"]
    chosenRandomSeed = int(hydraConfig["randomSeed"])
    chosenAqu = hydraConfig["aquisitionFunction"]
    chosenSurr = hydraConfig["surrogateFunction"]
    executeTraining = bool(hydraConfig["executeTraining"])
    saveResults = bool(hydraConfig["saveResults"])

    #Chosing Aquisition and Surrogate Function
    aquisitonFunctions = {
    "qExpectedImprovement":qExpectedImprovement,
    "qNoisyExpectedImprovement":qNoisyExpectedImprovement,
    "ExpectedImprovement":ExpectedImprovement,}
    surrogateFunctions = {"FixedNoiseGP":FixedNoiseGP, "SingleTaskGP":SingleTaskGP}

    #Chosing and initialising Benchmark
    if chosenBenchmark == "Branin":
        metric = BraninMetric(name="branin", param_names=["x1", "x2"])
        parameterlist = [
            RangeParameter(name="x1", parameter_type=ParameterType.FLOAT, lower=-5, upper=10),
            RangeParameter(name="x2", parameter_type=ParameterType.FLOAT, lower=0, upper=15),
        ]
    elif chosenBenchmark == "Hartmann6":
        metric = Hartmann6Metric(
            name="hartmann6", param_names=["x1", "x2", "x3", "x4", "x5", "x6"]
        )
        parameterlist = [
            RangeParameter(name="x1", parameter_type=ParameterType.FLOAT, lower=0, upper=1),
            RangeParameter(name="x2", parameter_type=ParameterType.FLOAT, lower=0, upper=1),
            RangeParameter(name="x3", parameter_type=ParameterType.FLOAT, lower=0, upper=1),
            RangeParameter(name="x4", parameter_type=ParameterType.FLOAT, lower=0, upper=1),
            RangeParameter(name="x5", parameter_type=ParameterType.FLOAT, lower=0, upper=1),
            RangeParameter(name="x6", parameter_type=ParameterType.FLOAT, lower=0, upper=1),
        ]
    elif chosenBenchmark == "Jahs_Bench":
        jahs_benchmark = jahs_bench.Benchmark(task="cifar10")

        class JAHSMetric(Metric):
            def fetch_trial_data(self, trial):
                records = []
                for arm_name, arm in trial.arms_by_name.items():
                    params = arm.parameters
                    nepochs = params.pop("epoch")
                    results = jahs_benchmark(params, nepochs)
                    records.append(
                        {
                            "arm_name": arm_name,
                            "metric_name": self.name,
                            "trial_index": trial.index,
                            "mean": 100-results[nepochs]["valid-acc"],
                            "sem": 0.0,
                        }
                    )
                return Ok(value=Data(df=pd.DataFrame.from_records(records)))

        metric = JAHSMetric("JAHS")

        parameterlist = [
            ChoiceParameter(
                "Activation",
                ParameterType.STRING,
                ["ReLU", "Hardswish", "Mish"],
                False,
                sort_values=False,
            ),
            RangeParameter("LearningRate", ParameterType.FLOAT, 0, 1e-3),
            RangeParameter("WeightDecay", ParameterType.FLOAT, 1e-5, 1e-2),
            ChoiceParameter(
                "TrivialAugment", ParameterType.BOOL, [True, False], True, sort_values=False
            ),
            ChoiceParameter(
                "Op1", ParameterType.INT, [0, 1, 2, 3, 4], False, sort_values=True
            ),
            ChoiceParameter(
                "Op2", ParameterType.INT, [0, 1, 2, 3, 4], False, sort_values=True
            ),
            ChoiceParameter(
                "Op3", ParameterType.INT, [0, 1, 2, 3, 4], False, sort_values=True
            ),
            ChoiceParameter(
                "Op4", ParameterType.INT, [0, 1, 2, 3, 4], False, sort_values=True
            ),
            ChoiceParameter(
                "Op5", ParameterType.INT, [0, 1, 2, 3, 4], False, sort_values=True
            ),
            ChoiceParameter(
                "Op6", ParameterType.INT, [0, 1, 2, 3, 4], False, sort_values=True
            ),
            ChoiceParameter(
                "N", ParameterType.INT, [1, 3, 5], sort_values=True, is_ordered=False
            ),
            ChoiceParameter(
                "W", ParameterType.INT, [4, 8, 16], sort_values=True, is_ordered=False
            ),
            ChoiceParameter(
                "Resolution",
                ParameterType.FLOAT,
                [0.25, 0.5, 1],
                is_ordered=False,
                sort_values=True,
            ),
            RangeParameter("epoch", ParameterType.INT, 1, 200),
            FixedParameter("Optimizer", ParameterType.STRING, "SGD"),
        ]
    else:
        try:
            raise KeyboardInterrupt
        finally:
            print("No valid benchmark")

    #HPO_Bench_NN_Integration: (uncomment Import to work + create Hydra-File. Problem: getuid()-Command used in PIP/Poetry-Import only works on Linux)
    '''
    elif chosenBenchmark == "NN_HPO_Bench":
        HPOnnBenchmark=NNBenchmark(task_id=1)

        class HPOBench_NN_Metric(Metric):
            def fetch_trial_data(self, trial):
                records = []
                for arm_name, arm in trial.arms_by_name.items():
                    params = arm.parameters
                    results = HPOnnBenchmark.objective_function(params)
                    records.append(
                        {
                            "arm_name": arm_name,
                            "metric_name": self.name,
                            "trial_index": trial.index,
                            "mean": results["function_value"],
                            "sem": 0,
                        }
                    )
                return Ok(value=Data(df=pd.DataFrame.from_records(records)))
            
        metric=HPOBench_NN_Metric("HPO_NN_metric")

        parameterlist=[
            RangeParameter("alpha",ParameterType.FLOAT,1e-08,1-0,True),
            RangeParameter("batch_size",ParameterType.INT,4,256,True),
            RangeParameter("depth",ParameterType.INT,1,3),
            RangeParameter("learning_rate_init",ParameterType.FLOAT,1e-05,1.0,True),
            RangeParameter("width",ParameterType.INT,16,1024,True)
        ]
    '''
    

    #Creating MockRunner-Class
    class MockRunner(Runner):
        @classmethod
        def run(cls, trial):
            return {"name": str(trial.index)}

    #Initialising Experiment
    exp = Experiment(
        name="experiment " + chosenAqu+" "+chosenSurr,
        search_space=SearchSpace(parameters=parameterlist),
        optimization_config=OptimizationConfig(objective=Objective(
                metric=metric,
                minimize=True,
            ),
        ),
        runner=MockRunner()
    )

    if executeTraining:
        #Sobol-Trainingrounds
        sobol = Models.SOBOL(exp.search_space)
        for i in range(numberOfSobolRounds):
            trial = exp.new_trial(generator_run=sobol.gen(1))
            trial.run()
            trial.mark_completed()

        #Botorch-Traningrounds
        for i in range(numberOfBotorchRounds):
            model_bridge_with_GPEI = Models.BOTORCH_MODULAR(
                experiment=exp,
                data=exp.fetch_data(),
                surrogate=Surrogate(surrogateFunctions[chosenSurr]),
                botorch_acqf_class=aquisitonFunctions[chosenAqu]
            )
            generator_run = model_bridge_with_GPEI.gen(1)
            #best_arm, _ = generator_run.best_arm_predictions
            trial = exp.new_trial(generator_run=generator_run)
            trial.run()
            trial.mark_completed()

        #Data processing
        df = exp.fetch_data().df
        df.insert(0, "surrogate+aquistion", str(chosenSurr+"+"+chosenAqu))
        df.insert(0, "benchmark", str(chosenBenchmark))
        df.insert(0, "seed", str(chosenRandomSeed))
        df = df.drop("arm_name", axis="columns")
        df = df.rename(columns={"trial_index": "budget"})
        print(df)

        #Data saving    
        if saveResults:
            os.makedirs("./results", exist_ok=True)
            df.to_pickle(
                "./results/b"
                + str(chosenBenchmark)
                + "a"
                + str(chosenAqu)
                + "s"
                + str(chosenSurr)
                + ".pkl"
            )

    print("finished")

if __name__ == "__main__":
    testdict={
        "sobolRounds":"5",
        "botorchRounds":"15",
        "benchmark":"Branin",
        "randomSeed":"0",
        "aquisitionFunction":"ExpectedImprovement",
        "surrogateFunction":"SingleTaskGP",
        "executeTraining":"False",
        "saveResults":"False"
    }
    runExperiment(testdict)
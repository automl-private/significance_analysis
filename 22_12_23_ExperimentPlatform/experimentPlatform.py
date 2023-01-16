import os

# import pandas as pd
from ax import (
    Experiment,
    Objective,
    OptimizationConfig,
    ParameterType,
    RangeParameter,
    Runner,
    SearchSpace,
)
from ax.metrics.branin import BraninMetric
from ax.metrics.hartmann6 import Hartmann6Metric
from ax.modelbridge.registry import Models  # , Cont_X_trans,Y_trans

# Ax wrappers for BoTorch components
# from ax.models.torch.botorch_modular.model import BoTorchModel
from ax.models.torch.botorch_modular.surrogate import Surrogate

# Test Ax objects
# from ax.utils.testing.core_stubs import (get_branin_data,get_branin_data_multi_objective,get_branin_experiment,get_branin_experiment_with_multi_objective,)
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
    qNoisyExpectedImprovement,
)
from botorch.models.gp_regression import FixedNoiseGP, SingleTaskGP

# Experiment examination utilities
# from ax.service.utils.report_utils import exp_to_df

# BoTorch components
# from botorch.models.model import Model
# from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

aquisitonFunctions = [
    qExpectedImprovement,
    qNoisyExpectedImprovement,
    ExpectedImprovement,
]
surrogateFunctions = [FixedNoiseGP, SingleTaskGP]

"""
TO-DO:
Get from terminal/hydra
"""
numberOfSobolRounds = 5
numberOfBotorchRounds = 5
chosenBenchmark = int(input("Which benchmark? "))
chosenRandomSeed = int(input("Which seed? "))
chosenAlgorithm = int(
    input(
        "Which algorithm? (Int from 0 to "
        + str(len(surrogateFunctions) * len(aquisitonFunctions))
        + ") "
    )
)

chosenAqu = int(chosenAlgorithm / len(surrogateFunctions))
chosenSurr = chosenAlgorithm % len(surrogateFunctions)
if chosenBenchmark == 0:
    metric = BraninMetric(name="branin", param_names=["x1", "x2"])
    parameterlist = [
        RangeParameter(name="x1", parameter_type=ParameterType.FLOAT, lower=-5, upper=10),
        RangeParameter(name="x2", parameter_type=ParameterType.FLOAT, lower=0, upper=15),
    ]
elif chosenBenchmark == 1:
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


class MockRunner(Runner):
    @classmethod
    def run(cls, trial):
        return {"name": str(trial.index)}


exp = Experiment(
    name="experiment" + str(chosenAlgorithm),
    search_space=SearchSpace(parameters=parameterlist),
    optimization_config=OptimizationConfig(
        objective=Objective(
            metric=metric,
            minimize=True,
        ),
    ),
    runner=MockRunner(),
)

sobol = Models.SOBOL(exp.search_space)
for i in range(numberOfSobolRounds):
    trial = exp.new_trial(generator_run=sobol.gen(1))
    trial.run()
    trial.mark_completed()


for i in range(numberOfBotorchRounds):
    model_bridge_with_GPEI = Models.BOTORCH_MODULAR(
        experiment=exp,
        data=exp.fetch_data(),
        surrogate=Surrogate(surrogateFunctions[chosenSurr]),
        botorch_acqf_class=aquisitonFunctions[chosenAqu],
    )
    generator_run = model_bridge_with_GPEI.gen(1)
    best_arm, _ = generator_run.best_arm_predictions
    trial = exp.new_trial(generator_run=generator_run)
    trial.run()
    trial.mark_completed()


df = exp.fetch_data().df
df.insert(0, "algorithm", str(chosenAlgorithm))
df.insert(0, "benchmark", str(chosenBenchmark))
df.insert(0, "seed", str(chosenRandomSeed))
df = df.drop("arm_name", axis="columns")
df = df.rename(columns={"trial_index": "budget"})
print(df)
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

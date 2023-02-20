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
from ax.modelbridge.registry import Models
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.utils.common.result import Ok
from botorch.acquisition.analytic import (  # ProbabilityOfImprovement,; UpperConfidenceBound,
    ExpectedImprovement,
)

# from botorch.acquisition.monte_carlo import (  # qNoisyExpectedImprovement,
# qExpectedImprovement,
# qProbabilityOfImprovement,
# qUpperConfidenceBound,
# )
# from botorch.models.gp_regression import (
# FixedNoiseGP,
# HeteroskedasticSingleTaskGP,
# SingleTaskGP,
# )
from botorch.models.gp_regression_mixed import MixedSingleTaskGP

jahs_benchmark = jahs_bench.Benchmark(task="cifar10")

# metric = Hartmann6Metric(
#    name="hartmann6", param_names=["x1", "x2", "x3", "x4", "x5", "x6"]
# )
parameterlist = [
    ChoiceParameter(
        "Activation",
        ParameterType.STRING,
        ["ReLU", "Hardswish", "Mish"],
        is_ordered=False,
        sort_values=False,
    ),
    RangeParameter("LearningRate", ParameterType.FLOAT, 0, 1e-3),
    RangeParameter("WeightDecay", ParameterType.FLOAT, 1e-5, 1e-2),
    ChoiceParameter(
        "TrivialAugment", ParameterType.BOOL, [True, False], True, sort_values=False
    ),
    ChoiceParameter("Op1", ParameterType.INT, [0, 1, 2, 3, 4], False, sort_values=True),
    ChoiceParameter("Op2", ParameterType.INT, [0, 1, 2, 3, 4], False, sort_values=True),
    ChoiceParameter("Op3", ParameterType.INT, [0, 1, 2, 3, 4], False, sort_values=True),
    ChoiceParameter("Op4", ParameterType.INT, [0, 1, 2, 3, 4], False, sort_values=True),
    ChoiceParameter("Op5", ParameterType.INT, [0, 1, 2, 3, 4], False, sort_values=True),
    ChoiceParameter("Op6", ParameterType.INT, [0, 1, 2, 3, 4], False, sort_values=True),
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


class JAHS_BENCH_Metric(Metric):
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
                    "mean": 100 - results[nepochs]["valid-acc"],
                    "sem": 0,
                }
            )
        return Ok(value=Data(df=pd.DataFrame.from_records(records)))


"""
    def keys(self):
        return [
            "Op1",
            "Op2",
            "Op3",
            "Op4",
            "Op5",
            "Op6",
            "Activation",
            "LearningRate",
            "WeightDecay",
            "TrivialAugment",
            "N",
            "W",
            "Resolution",
            "epoch",
            "Optimizer",
        ]
"""


class MockRunner(Runner):
    @classmethod
    def run(cls, trial):
        return {"name": str(trial.index)}


exp = Experiment(
    name="experiment_jahsbench",
    search_space=SearchSpace(parameters=parameterlist),
    optimization_config=OptimizationConfig(
        objective=Objective(
            metric=JAHS_BENCH_Metric("jahs_metric"),
            minimize=True,
        ),
    ),
    runner=MockRunner(),
)

sobol = Models.SOBOL(exp.search_space)
for i in range(1):
    trial = exp.new_trial(generator_run=sobol.gen(1))
    trial.run()
    trial.mark_completed()

# Botorch-Traningrounds
for _ in range(2):
    model_bridge_with_GPEI = Models.BOTORCH_MODULAR(
        experiment=exp,
        data=exp.fetch_data(),
        surrogate=Surrogate(
            MixedSingleTaskGP, model_options={"categorical_features": [0, 3, 4]}
        ),
        botorch_acqf_class=ExpectedImprovement,
    )
    generator_run = model_bridge_with_GPEI.gen(1)
    # best_arm, _ = generator_run.best_arm_predictions
    trial = exp.new_trial(generator_run=generator_run)
    trial.run()
    trial.mark_completed()

df = exp.fetch_data().df
print(df)


# Query a random configuration
# config = jahs_benchmark.sample_config()
# results=jahs_benchmark(config)


# print(
#    f"Result: {results}"
# )  # A dict of dicts, indexed first by epoch and then by metric name

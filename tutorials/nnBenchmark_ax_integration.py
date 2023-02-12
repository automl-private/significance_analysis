from hpobench.benchmarks.ml.nn_benchmark import NNBenchmark

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
from ax.utils.common.result import Ok




HPOnnBenchmark=NNBenchmark(task_id=1)


class HPOBench_NN_Metric(Metric):
    def fetch_trial_data(self, trial):
        records = []
        for arm_name, arm in trial.arms_by_name.items():
            params = arm.parameters
            results = HPOnnBenchmark.objective_function(params,)
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


parameterlist=[
    RangeParameter("alpha",ParameterType.FLOAT,1e-08,1-0,True),
    RangeParameter("batch_size",ParameterType.INT,4,256,True),
    RangeParameter("depth",ParameterType.INT,1,3),
    RangeParameter("learning_rate_init",ParameterType.FLOAT,1e-05,1.0,True),
    RangeParameter("width",ParameterType.INT,16,1024,True)
]


class MockRunner(Runner):
    @classmethod
    def run(cls, trial):
        return {"name": str(trial.index)}


exp = Experiment(
    name="experiment_HPOnnBench",
    search_space=SearchSpace(parameters=parameterlist),
    optimization_config=OptimizationConfig(
        objective=Objective(
            metric=HPOBench_NN_Metric("HPO_nn_metric"),
            minimize=True,
        ),
    ),
    runner=MockRunner(),
)

sobol = Models.SOBOL(exp.search_space)
for i in range(2):
    trial = exp.new_trial(generator_run=sobol.gen(1))
    trial.run()
    trial.mark_completed()
    df = exp.fetch_data().df
    print(df)


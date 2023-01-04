from ax.service.ax_client import AxClient
from ax.utils.measurement.synthetic_functions import branin
from typing import Any, Dict, Optional, Tuple, Type
from ax import *
from ax.metrics.branin import BraninMetric

# Ax wrappers for BoTorch components
from ax.models.torch.botorch_modular.model import BoTorchModel
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.models.torch.botorch_modular.list_surrogate import ListSurrogate
from ax.models.torch.botorch_modular.acquisition import Acquisition

# Ax data tranformation layer
from ax.modelbridge.torch import TorchModelBridge
from ax.modelbridge.registry import Cont_X_trans, Y_trans, Models

# Experiment examination utilities
from ax.service.utils.report_utils import exp_to_df

# Test Ax objects
from ax.utils.testing.core_stubs import (
    get_branin_experiment, 
    get_branin_data, 
    get_branin_experiment_with_multi_objective,
    get_branin_data_multi_objective,
)

# BoTorch components
from botorch.models.model import Model
from botorch.models.gp_regression import FixedNoiseGP,SingleTaskGP
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.acquisition.analytic import ExpectedImprovement
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

aquisitonFunctions=[qExpectedImprovement,qNoisyExpectedImprovement,ExpectedImprovement]
surrogateFunctions=[FixedNoiseGP,SingleTaskGP]

"""
TO-DO:
Get from terminal/hydra
"""
selectedAlgorithm=0
chosenAqu=aquisitonFunctions[int(selectedAlgorithm/len(surrogateFunctions))]
chosenSurr=surrogateFunctions[selectedAlgorithm%len(surrogateFunctions)]

"""
TO-DO:
Get from terminal/hydra
"""
numberOfSobolRounds=5
numberOfBotorchRounds=15


class MockRunner(Runner):
    def run(self, trial):
        return {"name": str(trial.index)}

"""
TO-DO:
Get from benchmark
"""
exp_search_space = SearchSpace(
    parameters=[
        RangeParameter(
            name="x1", parameter_type=ParameterType.FLOAT, lower=-5, upper=10
        ),
        RangeParameter(
            name="x2", parameter_type=ParameterType.FLOAT, lower=0, upper=15
        ),
    ]
)

"""
TO-DO:
Get metric from benchmark
"""
exp = Experiment(
    name="experiment"+str(selectedAlgorithm),
    search_space=exp_search_space,
    optimization_config=OptimizationConfig(
        objective=Objective(
            metric=BraninMetric(name="branin", param_names=["x1", "x2"]),
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

best_arm = None
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


print(exp.fetch_data().df)
#best_parameters = best_arm.parameters
#print(best_parameters)
print("finished")
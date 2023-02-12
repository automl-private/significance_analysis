from ax import (
    ComparisonOp,
    ParameterType, 
    RangeParameter,
    ChoiceParameter,
    FixedParameter,
    SearchSpace, 
    Experiment, 
    OutcomeConstraint, 
    OrderConstraint,
    SumConstraint,
    OptimizationConfig,
    Objective,
    Metric,
)
from ax.utils.notebook.plotting import render, init_notebook_plotting

init_notebook_plotting()

#Create Search Space of HPs
hartmann_search_space = SearchSpace(
    parameters=[
        RangeParameter(
            name=f"x{i}", parameter_type=ParameterType.FLOAT, lower=0.0, upper=1.0
        )
        for i in range(6)
    ]
)

#Parameters can also be choice and fixed:
#choice_param = ChoiceParameter(name="choice", values=["foo", "bar"], parameter_type=ParameterType.STRING)
#fixed_param = FixedParameter(name="fixed", value=[True], parameter_type=ParameterType.BOOL)

#Also sum (sum of params smaller than x) and order (param1 smaller param2) constraints can be applied

'''
sum_constraint = SumConstraint(
    parameters=[hartmann_search_space.parameters['x0'], hartmann_search_space.parameters['x1']], 
    is_upper_bound=True, 
    bound=5.0,
)

order_constraint = OrderConstraint(
    lower_parameter = hartmann_search_space.parameters['x0'],
    upper_parameter = hartmann_search_space.parameters['x1'],
)'''

from ax.metrics.l2norm import L2NormMetric
from ax.metrics.hartmann6 import Hartmann6Metric

#Create Optimization Config
#Minimizing the Hartmann-Objective, while constraining, that the l2norm<1.25
param_names = [f"x{i}" for i in range(6)]
optimization_config = OptimizationConfig(
    objective = Objective(
        metric=Hartmann6Metric(name="hartmann6", param_names=param_names), 
        minimize=True,
    ),
    outcome_constraints=[
        OutcomeConstraint(
            metric=L2NormMetric(
                name="l2norm", param_names=param_names, noise_sd=0.2
            ),
            op=ComparisonOp.LEQ,
            bound=1.25,
            relative=False,
        )
    ],
)

#Define Runner
from ax import Runner

class MyRunner(Runner):
    def run(self, trial):
        trial_metadata = {"name": str(trial.index)}
        return trial_metadata

#Create Experiment, using searchspace. optimizationconfig and runner
exp = Experiment(
    name="test_hartmann",
    search_space=hartmann_search_space,
    optimization_config=optimization_config,
    runner=MyRunner(),
)

#Perform HPO
from ax.modelbridge.registry import Models

NUM_SOBOL_TRIALS = 5
NUM_BOTORCH_TRIALS = 15

print(f"Running Sobol initialization trials...")
sobol = Models.SOBOL(search_space=exp.search_space)
    
for i in range(NUM_SOBOL_TRIALS):
    # Produce a GeneratorRun from the model, which contains proposed arm(s) and other metadata
    generator_run = sobol.gen(n=1)
    # Add generator run to a trial to make it part of the experiment and evaluate arm(s) in it
    trial = exp.new_trial(generator_run=generator_run)
    # Start trial run to evaluate arm(s) in the trial
    trial.run()
    # Mark trial as completed to record when a trial run is completed 
    # and enable fetching of data for metrics on the experiment 
    # (by default, trials must be completed before metrics can fetch their data,
    # unless a metric is explicitly configured otherwise)
    trial.mark_completed()

for i in range(NUM_BOTORCH_TRIALS):
    print(
        f"Running GP+EI optimization trial {i + NUM_SOBOL_TRIALS + 1}/{NUM_SOBOL_TRIALS + NUM_BOTORCH_TRIALS}..."
    )
    # Reinitialize GP+EI model at each step with updated data.
    gpei = Models.BOTORCH(experiment=exp, data=exp.fetch_data())
    generator_run = gpei.gen(n=1)
    trial = exp.new_trial(generator_run=generator_run)
    trial.run()
    trial.mark_completed()
    
print("Done!")

#Inspect trials data
trial_data = exp.fetch_trials_data([NUM_SOBOL_TRIALS + NUM_BOTORCH_TRIALS - 1])
print(trial_data.df)
print(exp.fetch_data().df)

#Plot results
import numpy as np
from ax.plot.trace import optimization_trace_single_method

# `plot_single_method` expects a 2-d array of means, because it expects to average means from multiple 
# optimization runs, so we wrap out best objectives array in another array.
objective_means = np.array([[trial.objective_mean for trial in exp.trials.values()]])
best_objective_plot = optimization_trace_single_method(
        y=np.minimum.accumulate(objective_means, axis=1),
        optimum=-3.32237,  # Known minimum objective for Hartmann6 function.
)
#render(best_objective_plot)


from hpobench.benchmarks.ml.nn_benchmark import NNBenchmark
from hpobench.dependencies.ml.ml_benchmark_template import MLBenchmark

from hpobench.container.benchmarks.nas.tabular_benchmarks import SliceLocalizationBenchmark

HPOmlBenchmark=MLBenchmark()
HPOnnBenchmark=NNBenchmark(HPOmlBenchmark)

print(HPOnnBenchmark.configuration_space)

b = SliceLocalizationBenchmark(rng=1)
config = b.get_configuration_space(seed=1).sample_configuration()
result_dict = b.objective_function(configuration=config, fidelity={"budget": 100}, rng=1)
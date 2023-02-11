from hpobench.benchmarks.ml.nn_benchmark import NNBenchmark
from hpobench.benchmarks.ml.svm_benchmark import SVMBenchmark


HPOsvmBenchmark=SVMBenchmark(task_id=1)
HPOnnBenchmark=NNBenchmark(task_id=2)

a=HPOnnBenchmark.get_config()
print("Config alpha: ",a, "type: ",type(a["alpha"]))
a_lib={'alpha': 4.636242428420627e-05,
  'batch_size': 27,
  'depth': 3,
  'learning_rate_init': 0.05172542125174575,
  'width': 920,
}
b=HPOnnBenchmark.objective_function(a_lib)
print(b["function_value"])
#c=HPOnnBenchmark.objective_function_test(a)
#print("Objective function",b)
#print("Objective test",c)
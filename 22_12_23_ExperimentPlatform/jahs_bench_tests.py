import jahs_bench

# import pandas

benchmark = jahs_bench.Benchmark(task="cifar10")

config = benchmark.sample_config()
print(config)
results = benchmark(config, nepochs=1)
print(results)

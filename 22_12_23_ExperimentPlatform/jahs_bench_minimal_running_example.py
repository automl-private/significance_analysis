import jahs_bench

jahs_benchmark = jahs_bench.Benchmark(task="cifar10")

config = jahs_benchmark.sample_config()
epoch = config.pop("epoch")
results = jahs_benchmark(config, nepochs=epoch)

print(f"Config: {config}")

print(f"Result: {results}")
print(results[epoch])

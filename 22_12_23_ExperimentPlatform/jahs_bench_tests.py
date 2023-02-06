import jahs_bench

benchmark = jahs_bench.Benchmark(task="cifar10", download=True)

# Query a random configuration
config = benchmark.sample_config()
results = benchmark(config, nepochs=1)

# Display the outputs
print(f"Config: {config}")  # A dict
print(
    f"Result: {results}"
)  # A dict of dicts, indexed first by epoch and then by metric name

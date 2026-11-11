# Experiments

![Test accuracy of centralized, decentralized training on CIFAR10](assets/experiments_comparison_plot.png)

The plot compares centralized training, FedAvg, and DecAvg on two topologies (Barabási–Albert and Erdős–Rényi), all using the `CNNMnist` model on non-IID MNIST. To reproduce it, run:

```bash
PYTHON_GIL=0 python main.py \
    --seed 123 \
    --output_directory logs/ \
    --num_parallel_clients 50 \
    --concurrent_runs 4 \
    --experiment_name_filter ".*(ER|Centralized).*"
```

To plot the results, use the *Comparison plot* in the [notebooks repository](https://github.com/DecentralizedLearning/notebooks).

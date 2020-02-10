import ray
import ray.tune as tune

ray.init()
tune.run_experiments({
    "my_experiment": {
        "run": "A3C",
        "env": "BipedalWalkerHardcore-v3",
        "stop": {"episode_reward_mean": 300},
        "config": {
            "use_pytorch": True,
            "sample_async": False,
            "num_gpus": 0,
            "num_workers": 11,
            "lr": tune.grid_search([0.0001, 0.001]),
            "sample_batch_size": tune.grid_search([10, 20, 40]),
            "lambda": tune.grid_search([0.9, 1.0, 1.1]),
        }
    }
})
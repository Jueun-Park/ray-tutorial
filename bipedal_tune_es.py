import random
import ray
from ray.tune import run, sample_from
from ray.tune.schedulers import PopulationBasedTraining

if __name__ == "__main__":
    pbt = PopulationBasedTraining(
        time_attr="time_total_s",
        metric="episode_reward_mean",
        mode="max",
        perturbation_interval=120,
        resample_probability=0.25,
        hyperparam_mutations={
                "l2_coeff": lambda: random.uniform(0.001, 0.05),  # 0.005
                "noise_stdev": lambda: random.uniform(0.01, 0.1),  # 0.02
                "episodes_per_batch": [1000, 2000],  # 1000
                "train_batch_size": [10000, 20000, 40000],  # 10000
                "eval_prob": lambda: random.uniform(0.001, 0.05), # 0.003
                # "return_proc_mode": "centered_rank",
                # "num_workers": 10,
                "stepsize": [0.005, 0.01, 0.05, 0.1, 0.05],  # 0.01
                # "observation_filter": "MeanStdFilter",
                # "noise_size": 250000000,
                # "report_length": 10,
        },
    )

    ray.init()
    run(
        "ES",
        name="pbt_bipedal_test",
        scheduler=pbt,
        num_samples= 5,
        config={
            "env": "BipedalWalkerHardcore-v3",
        },
    )

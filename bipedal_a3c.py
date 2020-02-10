import ray
import ray.rllib.agents.a3c as a3c
from ray.tune.logger import pretty_print

ray.init()
config = a3c.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
config["num_workers"] = 1
config["sample_async"] = False
config["use_pytorch"] = True
agent = a3c.A3CAgent(config=config, env="BipedalWalkerHardcore-v3")

for i in range(1000):
    result = agent.train()
    print(pretty_print(result))

    if i % 100 == 0:
        checkpoint = agent.save()
        print("checkpoint saved at", checkpoint)

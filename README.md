# Solving continuous control via Q-learning
Simple implementation of the Decoupled Q-networks (DecQN) agent in TensorFlow 2. The implementation is based on the Acme library and follows their overall design patterns with additional customization of the run loop and logging, as well as agent definition. This is a minimal re-implementation that streamlines agent definition and should yield the performance reported in our accompanying paper.


## Installation
1. Get the code and change directory:

   ```git clone https://github.com/tseyde/decqn.git && cd decqn```

2. Install conda environment and activate:

   ```conda env create -f decqn.yml && conda activate decqn && cd decqn```
   
3. Run an example experiment with DecQN:

   ```python3 run_experiment.py --algorithm=decqn --task=walker_walk```
   
## Method overview

<p align="center">
<img src="https://user-images.githubusercontent.com/6940395/196541125-6281deab-5c66-42fe-8c2f-000cb9664a6a.PNG" width="80%" />
</p>

**Main changes compared to DQN**:

1. Discretize continuous action space along each dimension by only considering bang-bang actions

2. Instead of enumerating the action space, add 1 value output per dimension per bin to the Q-network

3. Recover overall value function by choosing one output per action dimension and taking the mean

This assumes a linear value function decomposition and treats single-agent continuous control as a multi-agent discrete control problem. The key difference to the original DQN agent is the reduced number of output dimensions of the Q-network and the additional aggregation across action dimensions. The remaining structure of the original agent may be left unchanged.

## Reference

If you find our agent or code useful in your own research, please refer to our paper:
```
@article{seyde2022solving,
  title={Solving Continuous Control via Q-learning},
  author={Seyde, Tim and Werner, Peter and Schwarting, Wilko and Gilitschenski, Igor and Riedmiller, Martin and Rus, Daniela and Wulfmeier, Markus},
  journal={arXiv preprint arXiv:2210.12566},
  year={2022}
}
```

## Benchmark performance
Performance on a variety of tasks from the DeepMind Control Suite as well as MetaWorld.

### Feature observations
DecQN trained on feature observations in comparison to the D4PG and DMPO baseline agents.

![decqn-returns-features](https://user-images.githubusercontent.com/6940395/196541359-d240531e-b88e-489b-9c72-24908006e2e0.png)

### Pixel observations
DecQN trained on pixel observations in comparison to the DrQ-v2 and DreamerV2 baseline agents.

![decqn-returns-pixels](https://user-images.githubusercontent.com/6940395/196541392-ccfde727-996d-42a9-a7f2-6480bb250f8a.png)

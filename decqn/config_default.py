# Structure inspired by Dreamer (https://github.com/danijar/dreamer)

from misc import utils


class ConfigDefault:
    def __init__(self):     
        self.config = self._create_config()

    def _create_config(self):
        config = utils.AttrDict()

        # General
        config.task = "walker_walk"
        config.num_episodes = 1000
        config.num_steps = 1000 * 1000
        config.seed = 0

        config.algorithm = 'dqn'
        config.num_bins = 2

        # Agent
        config.layer_size_network = [512, 512]
        config.learning_rate = 1e-4
        config.clip_gradients = True
        config.clip_gradients_norm = 40.0
        config.epsilon = 0.1
        
        config.discount = 0.99
        config.batch_size = 256
        config.prefetch_size = 4
        config.target_update_period = 100
        config.min_replay_size = 1000
        config.max_replay_size = 1000000
        config.samples_per_insert = 32.0  # 32.0

        config.importance_sampling_exponent = 0.2
        config.priority_exponent = 0.6

        config.use_double_q = False
        config.use_residual = False

        config.action_repeat = 1

        # Visual
        config.num_pixels = 84
        config.pad_size = 4
        config.layer_size_bottleneck = 100

        # Adder
        config.adder_n_step = 3

        # Admin
        config.logdir = "./logdir"
        config.device = "local"
        config.debug = False
        
        return config
    
    def get_config(self):
        return self.config

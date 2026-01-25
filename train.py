import gymnasium as gym
import torch
import os
from argparse import ArgumentParser

from stable_baselines3.common.monitor import Monitor
from custom_recurrent_ppo import CustomRecurrentPPO
from sb3_contrib import RecurrentPPO
def parse_args():
    parser = ArgumentParser(description="Train RL agent on Sokoban environment with ConvLSTM or ConvAtt policies")

    # Global arguments
    parser.add_argument("--env", type=str, default="Sokoban-v0", help="Environment ID")
    parser.add_argument("--algo", type=str, default="RecurrentPPO", help="RL algorithm to use")
    parser.add_argument("--policy", type=str, default="ConvAttPolicy", help="Policy architecture to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    # Training arguments
    parser.add_argument("--total-timesteps", type=int, default=10_000_000, help="Total training timesteps")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate for the optimizer")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    
    # Config file path
    parser.add_argument("--config-file", type=str, default="convatt_config.yaml", help="Path to configuration file")

    # TensorBoard arguments
    parser.add_argument("--tensorboard-log", type=str, default="./tensorboard/", help="Path for TensorBoard logs")

    #Device argument
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training (cpu or cuda)")

    return parser.parse_args()

def main():
    args = parse_args()
    # Create Sokoban environment
    if args.env == "Sokoban-v0":
        from sokoban_wrapper import SokobanCompactWrapper
        from gymnasium.envs.registration import register

        # Register Sokoban environment if not already registered
        try:
            gym.make('Sokoban-v0')
        except:
            # Register the environment
            register(
                id='Sokoban-v0',
                entry_point='gym_sokoban.envs:SokobanEnv',
                kwargs={
                    'dim_room': (7, 7),
                    'max_steps': 120,
                    'num_boxes': 2,
                    'render_mode': 'rgb_array'
                }
            )

        env_registered = gym.make('Sokoban-v0', render_mode='rgb_array')
        env_registered = SokobanCompactWrapper(env_registered)
    else:
        raise ValueError(f"Unsupported environment: {args.env}")

    # Wrap with Monitor for logging and set random seed
    env_registered = Monitor(env_registered)
    obs, _ = env_registered.reset(seed=args.seed)

    # Load policy class
    if args.policy == "ConvAttPolicy":
        from convatt import ConvAttPolicy
        policy_class = ConvAttPolicy
    elif args.policy == "ConvLSTMPolicy":
        from convlstm import ConvLSTMPolicy
        policy_class = ConvLSTMPolicy
    else:
        raise ValueError(f"Unsupported policy architecture: {args.policy}")

    # Load policy configuration from YAML file
    if not os.path.isfile(args.config_file):
        raise ValueError(f"Config file not found: {args.config_file}")

    import yaml
    with open(args.config_file, 'r') as file:
        config_kwargs = yaml.safe_load(file)

    # Initialize Custom RecurrentPPO (uses SequenceAwareRolloutBuffer)
    if args.algo == "RecurrentPPO":
        model = RecurrentPPO(
            policy_class,
            env_registered,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            tensorboard_log=args.tensorboard_log,
            device=args.device,
            seed=args.seed,
            verbose=1,
            policy_kwargs=config_kwargs,
        )
    else:
        raise ValueError(f"Unsupported RL algorithm: {args.algo}")
    # Train the model
    print(f"\n{'='*80}")
    print(f"Starting training with {args.policy}")
    print(f"  Total timesteps: {args.total_timesteps}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Device: {args.device}")
    print(f"  Buffer: SequenceAwareRolloutBuffer (fixes state dimension matching)")
    print(f"{'='*80}\n")

    model.learn(total_timesteps=args.total_timesteps)

    # Save the model
    model_name = f"sokoban_{args.policy.lower()}_model"
    model.save(model_name)
    print(f"\nTraining complete! Model saved as: {model_name}.zip")

if __name__ == "__main__":
    main()

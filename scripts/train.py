from init import init; init()
import argparse

def main():
    parser = argparse.ArgumentParser(description='Train model with config file')
    parser.add_argument('--config_path', type=str, required=True, help='Path to a config file')
    args = parser.parse_args()
    
    from robo_utils.train_utils import default_train_loop
    from configs.training import Config
    from robo_utils import callbacks as callbacks
    
    config = Config.load(args.config_path)
    callback_names = getattr(config, 'callbacks')
    callback_list = [getattr(callbacks, name) for name in callback_names]
    default_train_loop(config, callback_list)

    import wandb
    wandb.finish()


if __name__ == '__main__':
    main()

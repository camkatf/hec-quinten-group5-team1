import argparse

def get_args():
    """Argument parser.
    Returns:
      Dictionary of arguments.
    """
    parser = argparse.ArgumentParser(description="Arguments for your component", 
                                     formatter_class=argparse.MetavarTypeHelpFormatter)

    parser.add_argument("--config_path", 
                        type=str, 
                        required=False, 
                        default='config.yml',
                        help="Path of the yaml file containing the evaluation process configuration.")

    args = vars(parser.parse_args())

    return args

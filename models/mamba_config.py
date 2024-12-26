import json
import os
import argparse

def mamba_args():
    parser = argparse.ArgumentParser()
    print("extracting arguments")
    ## Model Settings
    parser.add_argument("--input_dim", type=int, default=104)
    parser.add_argument("--output_dim", type=int, default=80)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--d_state", type=int, default=32)
    parser.add_argument("--d_conv", type=int, default=4)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--n_layers", type=int, default=1)

    args, _ = parser.parse_known_args()

    return args
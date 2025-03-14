import os
import torch
import argparse
import shutil
from collections import OrderedDict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_path", type=str, default="/home/data/zwanggy/2023/temp_rel_bias/poe_mixin_baseline",
                        help="the dir that contains ensemble model.")
    parser.add_argument("--file_name", type=str, default="pytorch_model.bin", )
    parser.add_argument("--retain_key", type=str, default="robust_model",
                        help="only retain the weights that begin with retain_key.")
    args = parser.parse_args()

    ensemble_path = os.path.join(args.clean_path, "ensemble_" + args.file_name)
    origin_path = os.path.join(args.clean_path, args.file_name)

    if os.path.exists(ensemble_path):
        print("already cleaned")
        exit()

    shutil.copyfile(origin_path, ensemble_path)

    model_state_dict = torch.load(origin_path, map_location="cpu")

    cleaned_model_state_dict = OrderedDict()
    for key, value in model_state_dict.items():
        keys = key.split(".", 1)
        if keys[0] == args.retain_key:
            cleaned_model_state_dict[keys[1]] = value

    torch.save(cleaned_model_state_dict, origin_path)
    print("finish")

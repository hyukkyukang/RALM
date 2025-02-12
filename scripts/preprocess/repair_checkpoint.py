import torch

MODEL_STATE_DICT_KEY = "state_dict"


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def repair_checkpoint(path):
    # Load the checkpoint
    ckpt = torch.load(path, weights_only=False)

    # Repair the checkpoint
    in_state_dict = ckpt[MODEL_STATE_DICT_KEY]
    pairings = [
        (src_key, remove_prefix(src_key, "_orig_mod."))
        for src_key in in_state_dict.keys()
    ]
    if all(src_key == dest_key for src_key, dest_key in pairings):
        print(f"No need to repair checkpoint {path}")
        return  # Do not write checkpoint if no need to repair!
    out_state_dict = {}
    for src_key, dest_key in pairings:
        print(f"{src_key}  ==>  {dest_key}")
        out_state_dict[dest_key] = in_state_dict[src_key]
    ckpt[MODEL_STATE_DICT_KEY] = out_state_dict

    # Make use_torch_compile to False
    ckpt["hyper_parameters"]["training"]["use_torch_compile"] = False

    # Save the checkpoint
    print(f"Saving checkpoint to {path}")
    torch.save(ckpt, path)


if __name__ == "__main__":
    checkpoint_path = "/mnt/md0/hkkang/retro/last.ckpt"
    repair_checkpoint(checkpoint_path)
    print("Done!")

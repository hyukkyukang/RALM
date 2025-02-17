from src.model.utils import repair_checkpoint

if __name__ == "__main__":
    checkpoint_path = "/root/RETRO/runs/lion_1e-3/last.ckpt"
    repair_checkpoint(checkpoint_path)
    print("Done!")

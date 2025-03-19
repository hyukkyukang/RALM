from src.model.utils import repair_checkpoint

if __name__ == "__main__":
    checkpoint_path = "/root/RETRO/runs/default/checkpoint-2025-03-18_12-16-56.ckpt"
    repair_checkpoint(checkpoint_path)
    print("Done!")

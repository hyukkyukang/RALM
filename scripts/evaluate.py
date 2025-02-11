import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import logging
from datetime import timedelta

import hkkang_utils.misc as misc_utils
import hydra
import lightning as L
import torch
from omegaconf import DictConfig

from src.dataset import ReLLamaDataModule
from src.model import ReLLamaLightningModule

logger = logging.getLogger("Evaluation")


@hydra.main(version_base=None, config_path="/root/RETRO/config", config_name="config")
def main(cfg: DictConfig) -> None:
    # Load trained model
    assert cfg.ckpt_path, "Please provide the path to the checkpoint"
    model = ReLLamaLightningModule.load_from_checkpoint(cfg.ckpt_path)

    # Load data module and model
    data_module = ReLLamaDataModule(cfg=cfg)
    data_module.prepare_data()

    # Create trainer
    trainer = L.Trainer(
        deterministic=True,
        accelerator="cuda",
        devices=torch.cuda.device_count(),
        strategy=L.pytorch.strategies.DDPStrategy(
            timeout=timedelta(hours=5), static_graph=True, gradient_as_bucket_view=True
        ),
    )

    # Evaluate
    trainer.test(model, datamodule=data_module)
    logger.info("Evaluation completed")
    return None


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    misc_utils.load_dotenv()
    main()

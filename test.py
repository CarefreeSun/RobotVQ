import os
import torch
import pytorch_lightning as pl

class StepCheckpointCallback(pl.Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Save a checkpoint file manually
        filepath = os.path.join(trainer.default_root_dir, f"step_checkpoint-step_{trainer.global_step}.ckpt")
        trainer.save_checkpoint(filepath)

# Your model remains the same
class SimpleModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log('train/loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)

    def train_dataloader(self):
        dataset = torch.utils.data.TensorDataset(torch.randn(100, 10), torch.randn(100, 1))
        return torch.utils.data.DataLoader(dataset, batch_size=10)

# Trainer configuration
trainer = pl.Trainer(
    callbacks=[StepCheckpointCallback()],  # Use the custom callback
    max_epochs=3,
    log_every_n_steps=1,
    default_root_dir='./'  # Ensure this points to a valid directory with write access
)

model = SimpleModel()
trainer.fit(model)

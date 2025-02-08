import torch
import torch.nn as nn
import torch.optim as optim
import lightning as pl
from .components import DictionaryExpert, GatingNetwork
from ..configs import ModelConfig


class MoELightning(pl.LightningModule):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        self.gating_net = GatingNetwork(config.input_dim, config.hidden_dim, config.num_experts)
        self.experts = nn.ModuleList([
            DictionaryExpert(config.input_dim, config.code_dim)
            for _ in range(config.num_experts)
        ])

    def forward(self, x):
        gating_probs = self.gating_net(x)
        recons = []
        for k in range(self.config.num_experts):
            recon_k, _ = self.experts[k](x)
            recons.append(recon_k)
        recon_stack = torch.stack(recons, dim=0)
        gating_probs_t = gating_probs.transpose(0, 1).unsqueeze(-1)
        mixture_recon = (recon_stack * gating_probs_t).sum(dim=0)
        return mixture_recon, recon_stack, gating_probs

    def training_step(self, batch, batch_idx):
        x = batch[0]
        mixture_recon, _, gating_probs = self(x)

        # Reconstruction loss
        recon_loss = 0.5 * ((mixture_recon - x) ** 2).mean()

        # L2 regularization
        reg_loss = sum((expert.dictionary ** 2).sum() for expert in self.experts)
        reg_loss *= self.config.lambda_reg

        loss = recon_loss + reg_loss

        # Log metrics
        self.log('train/total_loss', loss)
        self.log('train/recon_loss', recon_loss)
        self.log('train/reg_loss', reg_loss)

        # Log expert utilization
        expert_probs = gating_probs.mean(dim=0)
        for k in range(self.config.num_experts):
            self.log(f'train/expert_{k}_prob', expert_probs[k])

        return loss

    def on_train_epoch_end(self):
        # Log learning rate
        opt = self.optimizers()
        if isinstance(opt, torch.optim.Adam):
            lr = opt.param_groups[0]['lr']
            self.log('train/learning_rate', lr)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.config.learning_rate)
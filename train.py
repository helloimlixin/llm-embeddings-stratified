import os
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
from sklearn.decomposition import PCA
import plotly.express as px
import pandas as pd

from data.sentiment_datamodule import MultiDomainSentimentDataModule
from models.moe import MoELightning


@hydra.main(config_path="configs", config_name="config")
def train_moe_model(cfg: DictConfig):
    # Print configuration
    print(OmegaConf.to_yaml(cfg))

    # Initialize wandb logger
    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.experiment_name,
        log_model=cfg.wandb.log_model
    )

    # Initialize data module
    data_module = MultiDomainSentimentDataModule(cfg.data)

    # Initialize model
    model = MoELightning(cfg.model)

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(cfg.trainer.default_root_dir, cfg.experiment_name),
        filename=f'{cfg.experiment_name}-{{epoch:02d}}-{{train_loss:.2f}}',
        monitor='train/total_loss',
        mode='min'
    )

    # Initialize trainer with wandb logger
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        callbacks=[checkpoint_callback],
        logger=wandb_logger,
        default_root_dir=cfg.trainer.default_root_dir
    )

    # Train model
    trainer.fit(model, data_module)

    # Visualization after training
    model.eval()
    with torch.no_grad():
        emb_t = data_module.X_tensor.to(model.device)
        _, _, gating_probs = model(emb_t)
        strata = gating_probs.argmax(dim=1).cpu().numpy()

    # PCA for visualization
    pca_3d = PCA(n_components=3)
    emb_3d = pca_3d.fit_transform(data_module.X_tensor.numpy())

    # Create visualization dataframe
    domain_label = [f"{d}_{l}" for d, l in zip(data_module.domains, data_module.labels)]
    df_plot = pd.DataFrame({
        "x": emb_3d[:, 0],
        "y": emb_3d[:, 1],
        "z": emb_3d[:, 2],
        "domain": data_module.domains,
        "label": data_module.labels,
        "stratum": strata,
        "domain_label": domain_label
    })

    # Create plot
    fig = px.scatter_3d(
        df_plot,
        x="x", y="y", z="z",
        color="domain_label",
        hover_data=["domain", "label", "stratum"],
        title=f"Multi-Domain Embeddings + MoE - {cfg.experiment_name}",
        width=800,
        height=600
    )

    # Save plot in experiment directory
    plot_path = os.path.join(cfg.trainer.default_root_dir,
                             cfg.experiment_name,
                             "visualization.html")
    fig.write_html(plot_path)

    # Log plot to wandb
    wandb.log({"embedding_visualization": wandb.Html(plot_path)})

    # Log expert usage statistics
    expert_usage = gating_probs.mean
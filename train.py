import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import plotly.express as px
from datasets import load_dataset, concatenate_datasets, Dataset
from huggingface_hub import hf_hub_download
from datasets.features import Value
import openai
import hydra
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field
from typing import Optional
from omegaconf import DictConfig, OmegaConf
import os
import logging
from pathlib import Path


# set OpenAI API key
openai.api_key = "OPENAI_API_KEY"


#############################################################################
# 1. Configuration Classes
#############################################################################
@dataclass
class WandbConfig:
    project: str = "moe-sentiment"
    entity: Optional[str] = None
    log_model: bool = True

@dataclass
class DataConfig:
    samples_per_domain: int = 50
    batch_size: int = 16
    pca_dim: int = 64
    embedding_model: str = "text-embedding-ada-002"


@dataclass
class ModelConfig:
    input_dim: int = 64
    hidden_dim: int = 128
    code_dim: int = 32
    num_experts: int = 3
    learning_rate: float = 1e-3
    lambda_reg: float = 1e-5


@dataclass
class TrainerConfig:
    max_epochs: int = 100
    accelerator: str = "auto"
    devices: int = 1
    default_root_dir: str = "experiments"
    log_every_n_steps: int = 1


@dataclass
class Config:
    wandb: WandbConfig = field(default_factory=WandbConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    experiment_name: str = "moe_default"


cs = ConfigStore.instance()
cs.store(name="config", node=Config)


#############################################################################
# 2. Data Module
#############################################################################
class MultiDomainSentimentDataModule(pl.LightningDataModule):
    def __init__(self, config: DataConfig):
        super().__init__()
        self.config = config
        self.EMBEDDING_MODEL = config.embedding_model
        self.cache_dir = Path.home() / '.cache' / 'huggingface' / 'datasets'
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download_and_load_imdb(self):
        """Download and load IMDB dataset."""
        try:
            # Try using the newer streaming approach
            print("Trying streaming approach for IMDB...")
            dataset = load_dataset(
                "imdb",
                streaming=True,
                split="train"
            )
            data = []
            for i, example in enumerate(dataset):
                if i >= 2000:
                    break
                data.append({
                    "text": example["text"],
                    "label": example["label"]
                })
            return Dataset.from_list(data)
        except Exception as e:
            print(f"Streaming approach failed: {e}")
            try:
                print("Trying direct file download for IMDB...")
                # Fallback to direct file download
                file_path = hf_hub_download(
                    repo_id="imdb",
                    filename="train.csv",
                    repo_type="dataset"
                )
                df = pd.read_csv(file_path)
                return Dataset.from_pandas(df.head(2000))
            except Exception as e2:
                print(f"Direct download failed: {e2}")
                # Create synthetic data for testing
                print("Creating synthetic IMDB data for testing...")
                return self._create_synthetic_data(2000, "imdb")

    def download_and_load_rotten_tomatoes(self):
        """Download and load Rotten Tomatoes dataset."""
        try:
            print("Trying streaming approach for Rotten Tomatoes...")
            dataset = load_dataset(
                "rotten_tomatoes",
                streaming=True,
                split="train"
            )
            data = []
            for i, example in enumerate(dataset):
                if i >= 2000:
                    break
                data.append({
                    "text": example["text"],
                    "label": example["label"]
                })
            return Dataset.from_list(data)
        except Exception as e:
            print(f"Streaming approach failed: {e}")
            # Create synthetic data for testing
            print("Creating synthetic Rotten Tomatoes data for testing...")
            return self._create_synthetic_data(2000, "rotten")

    def download_and_load_amazon(self):
        """Download and load Amazon Polarity dataset."""
        try:
            print("Trying streaming approach for Amazon Polarity...")
            dataset = load_dataset(
                "amazon_polarity",
                streaming=True,
                split="train"
            )
            data = []
            for i, example in enumerate(dataset):
                if i >= 5000:
                    break
                text = f"{example['title']} {example['content']}"
                data.append({
                    "text": text.strip(),
                    "label": example["label"]
                })
            return Dataset.from_list(data)
        except Exception as e:
            print(f"Streaming approach failed: {e}")
            # Create synthetic data for testing
            print("Creating synthetic Amazon data for testing...")
            return self._create_synthetic_data(5000, "amazon")

    def _create_synthetic_data(self, num_samples, domain):
        """Create synthetic data for testing purposes."""
        texts = [f"This is a synthetic {domain} review {i}" for i in range(num_samples)]
        labels = np.random.randint(0, 2, size=num_samples)
        return Dataset.from_dict({
            "text": texts,
            "label": labels
        })

    def unify_dataset(self, ds, domain_name):
        """Unify dataset format across different sources."""
        # Convert to dict if needed
        if hasattr(ds, 'to_dict'):
            ds = ds.to_dict()

        # Convert to Dataset if needed
        if not hasattr(ds, 'map'):
            ds = Dataset.from_dict(ds)

        # Ensure text and label columns exist
        if 'text' not in ds.column_names or 'label' not in ds.column_names:
            raise ValueError(f"Dataset must have 'text' and 'label' columns. Found: {ds.column_names}")

        # Keep only necessary columns
        keep_cols = ["text", "label"]
        remove_cols = [c for c in ds.column_names if c not in keep_cols]
        ds = ds.remove_columns(remove_cols)

        # Convert label to int
        ds = ds.map(lambda x: {"label": int(x["label"])})
        ds = ds.cast_column("label", Value("int64"))

        # Select samples and add domain
        ds_small = ds.select(range(min(self.config.samples_per_domain, len(ds))))
        ds_small = ds_small.add_column("domain", [domain_name] * len(ds_small))

        return ds_small

    def gpt3_embed_text(self, text):
        """Get GPT-3 embeddings for a piece of text."""
        if not text or text.strip() == "":
            return np.zeros(1536, dtype=np.float32)
        response = openai.embeddings.create(
            model=self.EMBEDDING_MODEL,
            input=text
        )
        emb = response.data[0].embedding
        return np.array(emb, dtype=np.float32)

    def setup(self, stage=None):
        """Set up the datasets."""
        try:
            # Load datasets with fallback mechanisms
            print("Loading IMDB dataset...")
            imdb_raw = self.download_and_load_imdb()

            print("Loading Rotten Tomatoes dataset...")
            rt_raw = self.download_and_load_rotten_tomatoes()

            print("Loading Amazon Polarity dataset...")
            ap_raw = self.download_and_load_amazon()

            # Unify datasets
            print("Unifying datasets...")
            imdb_ds = self.unify_dataset(imdb_raw, "imdb")
            rt_ds = self.unify_dataset(rt_raw, "rotten")
            ap_ds = self.unify_dataset(ap_raw, "amazon")

            # Combine datasets
            print("Combining datasets...")
            combined = concatenate_datasets([imdb_ds, rt_ds, ap_ds])

            # Get embeddings
            print("Getting embeddings...")
            all_embeddings = []
            for idx, txt in enumerate(combined["text"]):
                if idx % 50 == 0:
                    print(f"Processing embedding {idx}/{len(combined)}")
                emb = self.gpt3_embed_text(txt)
                all_embeddings.append(emb)
            all_embeddings = np.stack(all_embeddings, axis=0)

            # PCA reduction
            print("Performing PCA...")
            pca = PCA(n_components=self.config.pca_dim)
            emb_reduced = pca.fit_transform(all_embeddings)

            # Create tensor dataset
            self.X_tensor = torch.tensor(emb_reduced, dtype=torch.float32)
            self.dataset = TensorDataset(self.X_tensor)

            # Store metadata for visualization
            self.domains = combined["domain"]
            self.labels = combined["label"]

            print(f"Setup complete. Dataset size: {len(self.dataset)}")

        except Exception as e:
            print(f"Error in setup: {str(e)}")
            raise RuntimeError(f"Failed to set up data module: {str(e)}")

    def train_dataloader(self):
        return DataLoader(self.dataset,
                          batch_size=self.config.batch_size,
                          shuffle=True,
                          num_workers=4,
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.dataset,
                          batch_size=self.config.batch_size,
                          shuffle=False,
                          num_workers=4)


#############################################################################
# 3. Model Components
#############################################################################
class DictionaryExpert(nn.Module):
    def __init__(self, input_dim, code_dim):
        super().__init__()
        self.dictionary = nn.Parameter(torch.randn(input_dim, code_dim) * 0.01)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2 * code_dim),
            nn.ReLU(),
            nn.Linear(2 * code_dim, code_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = z @ self.dictionary.T
        return recon, z


class GatingNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, K):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, K)
        )

    def forward(self, x):
        logits = self.net(x)
        probs = torch.softmax(logits, dim=1)
        return probs


#############################################################################
# 4. Lightning Module
#############################################################################
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
        mixture_recon, _, _ = self(x)

        # Reconstruction loss
        recon_loss = 0.5 * ((mixture_recon - x) ** 2).mean()

        # L2 regularization
        reg_loss = sum((expert.dictionary ** 2).sum() for expert in self.experts)
        reg_loss *= self.config.lambda_reg

        loss = recon_loss + reg_loss
        self.log('train_loss', loss,
                 on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        mixture_recon, _, _ = self(x)

        # Reconstruction loss
        recon_loss = 0.5 * ((mixture_recon - x) ** 2).mean()

        # L2 regularization
        reg_loss = sum((expert.dictionary ** 2).sum() for expert in self.experts)
        reg_loss *= self.config.lambda_reg

        loss = recon_loss + reg_loss
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.config.learning_rate)


#############################################################################
# 5. Training Pipeline with Hydra
#############################################################################
@hydra.main(config_path="configs", config_name="config", version_base="1.1")
def train_moe_model(cfg: DictConfig):
    # Print configuration
    print(OmegaConf.to_yaml(cfg))

    # Initialize data module
    data_module = MultiDomainSentimentDataModule(cfg.data)

    # Initialize model
    model = MoELightning(cfg.model)

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(cfg.trainer.default_root_dir, cfg.experiment_name),
        filename=f'{cfg.experiment_name}-{{epoch:02d}}-{{train_loss:.2f}}',
        monitor='train_loss',
        mode='min'
    )

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        callbacks=[checkpoint_callback],
        default_root_dir=cfg.trainer.default_root_dir,
        log_every_n_steps=cfg.trainer.log_every_n_steps
    )

    # Train model
    torch.set_float32_matmul_precision('medium')
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

    # Create and save plot
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


if __name__ == "__main__":
    train_moe_model()

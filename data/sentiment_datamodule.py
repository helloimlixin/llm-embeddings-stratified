import lightning as pl
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
from datasets import load_dataset, concatenate_datasets
from datasets.features import Value
from sklearn.decomposition import PCA
import openai
from ..configs import DataConfig


class MultiDomainSentimentDataModule(pl.LightningDataModule):
    def __init__(self, config: DataConfig):
        super().__init__()
        self.config = config
        self.EMBEDDING_MODEL = config.embedding_model

    def unify_dataset(self, ds, domain_name):
        keep_cols = ["text", "label"]
        remove_cols = [c for c in ds.column_names if c not in keep_cols]
        ds = ds.remove_columns(remove_cols)
        ds = ds.map(lambda x: {"label": int(x["label"])})
        ds = ds.cast_column("label", Value("int64"))
        ds_small = ds.select(range(min(self.config.samples_per_domain, len(ds))))
        ds_small = ds_small.add_column("domain", [domain_name] * len(ds_small))
        return ds_small

    def gpt3_embed_text(self, text):
        if not text or text.strip() == "":
            return np.zeros(1536, dtype=np.float32)
        response = openai.embeddings.create(
            model=self.EMBEDDING_MODEL,
            input=text
        )
        emb = response.data[0].embedding
        return np.array(emb, dtype=np.float32)

    def setup(self, stage=None):
        # Load datasets
        imdb_raw = load_dataset("imdb", split="train[:2000]")
        rt_raw = load_dataset("rotten_tomatoes", split="train[:2000]")
        ap_raw = load_dataset("amazon_polarity", split="train[:5000]")

        # Process Amazon Polarity
        keep_cols = ["title", "content", "label"]
        drop_cols = [c for c in ap_raw.column_names if c not in keep_cols]
        ap_raw = ap_raw.remove_columns(drop_cols)

        def combine_text(example):
            txt = ""
            if example.get("title"):
                txt += example["title"] + " "
            if example.get("content"):
                txt += example["content"]
            return {"text": txt.strip()}

        ap_raw = ap_raw.map(combine_text)
        ap_raw = ap_raw.remove_columns(["title", "content"])

        # Unify datasets
        imdb_ds = self.unify_dataset(imdb_raw, "imdb")
        rt_ds = self.unify_dataset(rt_raw, "rotten")
        ap_ds = self.unify_dataset(ap_raw, "amazon")

        # Combine datasets
        combined = concatenate_datasets([imdb_ds, rt_ds, ap_ds])

        # Get embeddings
        all_embeddings = []
        for txt in combined["text"]:
            emb = self.gpt3_embed_text(txt)
            all_embeddings.append(emb)
        all_embeddings = np.stack(all_embeddings, axis=0)

        # PCA reduction
        pca = PCA(n_components=self.config.pca_dim)
        emb_reduced = pca.fit_transform(all_embeddings)

        # Create tensor dataset
        self.X_tensor = torch.tensor(emb_reduced, dtype=torch.float32)
        self.dataset = TensorDataset(self.X_tensor)

        # Store metadata for visualization
        self.domains = combined["domain"]
        self.labels = combined["label"]

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.config.batch_size)
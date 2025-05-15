""" Sequence model for RNA using Transformer """

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Dropout, TransformerEncoder, TransformerEncoderLayer
from torch_geometric.nn import global_mean_pool

from rnaglib.utils.misc import tonumpy


class SequenceModel(torch.nn.Module):
    @classmethod
    def from_task(cls, 
                  task, 
                  num_node_features=None, 
                  num_classes=None,
                  graph_level=None, 
                  multi_label=None,
                  **model_args):
        """ Create a model based on task metadata. """
        if num_node_features is None:
            num_node_features = task.metadata["num_node_features"]
        if num_classes is None:
            num_classes = task.metadata["num_classes"]
        if graph_level is None:
            graph_level = task.metadata["graph_level"]
        if multi_label is None:
            multi_label = task.metadata["multi_label"]

        activation = 'softmax' if num_classes > 2 else 'sigmoid'

        return cls(
            num_node_features=num_node_features,
            num_classes=num_classes,
            graph_level=graph_level,
            multi_label=multi_label,
            final_activation=activation,
            **model_args
        )

    def __init__(
        self,
        num_node_features,
        num_classes,
        graph_level=False,
        num_layers=2,
        hidden_channels=128,
        dropout_rate=0.5,
        multi_label=False,
        final_activation="sigmoid",
        num_heads=8,
        device=None
    ):
        super().__init__()
        self.num_node_features = num_node_features
        self.num_classes = num_classes
        self.graph_level = graph_level
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.dropout_rate = dropout_rate
        self.multi_label = multi_label
        self.num_heads = num_heads

        # Input embedding layer
        self.input_embedding = torch.nn.Linear(num_node_features, hidden_channels)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_channels, dropout_rate)

        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_channels,
            nhead=num_heads,
            dim_feedforward=hidden_channels * 4,
            dropout=dropout_rate,
            activation="relu"
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Batch normalization and dropout
        self.bn = BatchNorm1d(hidden_channels)
        self.dropout = Dropout(dropout_rate)

        # Final activation
        if final_activation == "sigmoid":
            self.final_activation = torch.nn.Sigmoid()
        elif final_activation == "softmax":
            self.final_activation = torch.nn.Softmax(dim=1)
        else:
            self.final_activation = torch.nn.Identity()

        # Output layer
        if self.multi_label:
            self.final_linear = torch.nn.Linear(hidden_channels, num_classes)
            self.criterion = torch.nn.BCEWithLogitsLoss()
            self.final_activation = torch.nn.Identity()
        elif num_classes == 2:
            self.final_linear = torch.nn.Linear(hidden_channels, 1)
            self.criterion = torch.nn.BCEWithLogitsLoss()
            self.final_activation = torch.nn.Identity()
        else:
            self.final_linear = torch.nn.Linear(hidden_channels, num_classes)
            self.criterion = torch.nn.CrossEntropyLoss()
            if final_activation == "sigmoid":
                self.final_activation = torch.nn.Sigmoid()
            elif final_activation == "softmax":
                self.final_activation = torch.nn.Softmax(dim=1)
            else:
                self.final_activation = torch.nn.Identity()

        self.optimizer = None
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.configure_training()

    def forward(self, data):
        x, batch, chain_index = data.x, data.batch, data.chain_index  # Added chain_index

        # Embed residue types
        x = self.input_embedding(x)  # [num_nodes, hidden_channels]

        # Reshape for Transformer: [seq_len, batch_size, hidden_channels]
        seq_lengths = torch.bincount(batch)
        max_seq_len = seq_lengths.max().item()
        batch_size = seq_lengths.size(0)

        # Pad sequences to max length
        x_padded = torch.zeros(max_seq_len, batch_size, self.hidden_channels, device=x.device)
        mask = torch.ones(max_seq_len, batch_size, device=x.device, dtype=torch.bool)
        chain_index_padded = torch.zeros(max_seq_len, batch_size, dtype=torch.long, device=x.device)
        node_idx = 0
        for b in range(batch_size):
            seq_len = seq_lengths[b]
            x_padded[:seq_len, b] = x[node_idx:node_idx + seq_len]
            chain_index_padded[:seq_len, b] = chain_index[node_idx:node_idx + seq_len]
            mask[:seq_len, b] = False  # False for valid positions
            node_idx += seq_len

        # Apply positional encoding with chain_index
        x_padded = self.pos_encoder(x_padded, chain_index_padded)

        # Transformer expects [seq_len, batch_size, hidden_channels]
        x_transformed = self.transformer_encoder(x_padded, mask=None)  # [seq_len, batch_size, hidden_channels]

        # Reshape back to [num_nodes, hidden_channels]
        x_out = []
        for b in range(batch_size):
            seq_len = seq_lengths[b]
            x_out.append(x_transformed[:seq_len, b])
        x = torch.cat(x_out, dim=0)

        # Apply batch norm and dropout
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Pool for graph-level tasks
        if self.graph_level:
            x = global_mean_pool(x, batch)

        # Final linear layer and activation
        x = self.final_linear(x)
        x = self.final_activation(x)
        return x

    def configure_training(self, learning_rate=0.001):
        """Configure training settings."""
        self.to(self.device)
        self.criterion = self.criterion.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def compute_loss(self, out, target):
        if self.multi_label:
            target = target.float()
        elif self.num_classes == 2:
            out = out.flatten()
        loss = self.criterion(out, target)
        return loss

    def train_model(self, task, epochs=500):
        if self.optimizer is None:
            self.configure_training()

        if self.num_classes == 2:
            neg_count = float(task.metadata["class_distribution"]["0"])
            pos_count = float(task.metadata["class_distribution"]["1"])
            pos_weight = torch.tensor(np.sqrt(neg_count / pos_count)).to(self.device)
            self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        for epoch in range(epochs):
            self.train()
            epoch_loss = 0
            num_batches = 0
            for batch in task.train_dataloader:
                graph = batch["graph"].to(self.device)
                self.optimizer.zero_grad()
                out = self(graph)
                loss = self.compute_loss(out, graph.y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1

            if epoch % 10 == 0:
                val_metrics = self.evaluate(task, split="val")
                print(
                    f"Epoch {epoch}: train_loss = {epoch_loss / num_batches:.4f}, val_loss = {val_metrics['loss']:.4f}",
                )

    def inference(self, loader) -> tuple:
        self.eval()
        all_probs = []
        all_preds = []
        all_labels = []
        total_loss = 0
        with torch.no_grad():
            for batch in loader:
                graph = batch["graph"]
                graph = graph.to(self.device)
                out = self(graph)
                labels = graph.y
                loss = self.compute_loss(out, labels)
                total_loss += loss.item()

                preds = (out > 0).float() if (self.multi_label or self.num_classes == 2) else out.argmax(dim=1)
                probs = out

                probs = tonumpy(probs)
                preds = tonumpy(preds)
                labels = tonumpy(labels)

                if not self.graph_level:
                    cumulative_sizes = tuple(tonumpy(graph.ptr))
                    probs = [
                        probs[start:end]
                        for start, end in zip(cumulative_sizes[:-1], cumulative_sizes[1:], strict=False)
                    ]
                    preds = [
                        preds[start:end]
                        for start, end in zip(cumulative_sizes[:-1], cumulative_sizes[1:], strict=False)
                    ]
                    labels = [
                        labels[start:end]
                        for start, end in zip(cumulative_sizes[:-1], cumulative_sizes[1:], strict=False)
                    ]
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(labels)

        if self.graph_level:
            all_probs = np.stack(all_probs)
            all_preds = np.stack(all_preds)
            all_labels = np.stack(all_labels)
        mean_loss = total_loss / len(loader)
        return mean_loss, all_preds, all_probs, all_labels

    def get_dataloader(self, task, split="test"):
        if split == "test":
            dataloader = task.test_dataloader
        elif split == "val":
            dataloader = task.val_dataloader
        else:
            dataloader = task.train_dataloader
        return dataloader

    def evaluate(self, task, split="test"):
        dataloader = self.get_dataloader(task=task, split=split)
        mean_loss, all_preds, all_probs, all_labels = self.inference(loader=dataloader)
        metrics = task.compute_metrics(all_preds, all_probs, all_labels)
        metrics["loss"] = mean_loss
        return metrics


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.d_model = d_model
        self.max_len = max_len

        # Precompute sinusoidal encoding for max_len positions
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x, chain_index=None):
        """
        Args:
            x: Tensor of shape [seq_len, batch_size, d_model]
            chain_index: Tensor of shape [seq_len, batch_size] indicating chain index for each residue
        """
        seq_len, batch_size, _ = x.shape

        if chain_index is not None:
            # Compute chain-relative positions
            positions = torch.zeros(seq_len, batch_size, device=x.device, dtype=torch.long)
            for b in range(batch_size):
                # Get chain indices for this batch
                chain_ids = chain_index[:, b]
                # Compute position within each chain
                for chain_id in chain_ids.unique():
                    mask = (chain_ids == chain_id) & (chain_ids != -1)  # Exclude padding (-1)
                    positions[mask, b] = torch.arange(mask.sum(), device=x.device)
        else:
            # Fallback to absolute positions
            positions = torch.arange(seq_len, device=x.device).unsqueeze(1).expand(-1, batch_size)

        # Apply positional encoding
        x = x + self.pe[positions, :].to(x.device)
        return self.dropout(x)

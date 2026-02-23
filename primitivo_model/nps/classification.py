import pandas as pd
import torch
from torch import nn

from primitivo_model.nps.criteria import CriteriaTaskLoader


class NPClassifier(nn.Module):
    """
    Neural Process-based Classifier (merged architecture).

    Uses a pre-trained Neural Process to extract representations via hooks,
    then applies temporal pooling and a classification head.

    Args:
        np_model: Pre-trained Neural Process model
        input_dim: Dimension of the representation. If None, automatically detected from UNet.
        num_classes: Number of output classes (default 2 for binary classification)
        hidden_dims: List of hidden layer dimensions for MLP classifier
        pooling: How to aggregate temporal dimension ('last', 'avg', 'max', 'attention')
        dropout: Dropout probability
        freeze_np: Whether to freeze the NP weights during training
        class_weights: Optional weights for each class in loss function
    """

    def __init__(
        self,
        np_model,
        hidden_dims=[256, 128],
        pooling="last",
        dropout=0.3,
        freeze_np=True,
        num_classes=2,
        class_weights=None,
    ):
        super().__init__()
        self.np_model = np_model
        self.pooling = pooling
        self.num_classes = num_classes

        # Automatically detect input dimension from UNet's final_linear layer
        unet = self.np_model.decoder[0]
        if not hasattr(unet.final_linear, "in_channels"):
            raise ValueError(
                "Could not access 'in_channels' from UNet final_linear layer. "
                "Please specify input_dim explicitly."
            )
        self.input_dim = unet.final_linear.in_channels

        # Optionally freeze the Neural Process weights
        if freeze_np:
            for param in self.np_model.parameters():
                param.requires_grad = False
        else:
            pass  # Weights will be fine-tuned

        # Temporal pooling (attention mechanism if selected)
        if pooling == "attention":
            self.attention = nn.Sequential(nn.Linear(self.input_dim, 1), nn.Softmax(dim=-1))

        # Build MLP classifier
        layers = []
        prev_dim = self.input_dim

        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

        # Loss function
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

    def pool_temporal(self, x):
        """
        Aggregate representation over temporal dimension.

        Args:
            x: Tensor of shape [batch, channels, time]
        Returns:
            pooled: Tensor of shape [batch, channels]
        """
        if self.pooling == "last":
            return x[:, :, -1]
        elif self.pooling == "avg":
            return x.mean(dim=-1)
        elif self.pooling == "max":
            return x.max(dim=-1)[0]
        elif self.pooling == "attention":
            # Transpose to [batch, time, channels]
            x_t = x.transpose(1, 2)
            # Compute attention weights: [batch, time, 1]
            attn_weights = self.attention(x_t)
            # Apply attention: [batch, time, channels] * [batch, time, 1] -> [batch, channels]
            return (x_t * attn_weights).sum(dim=1)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

    def forward(self, contexts, xt):
        """
        Args:
            contexts: Context observations (as in NP)
            xt: Target inputs (as in NP)
        Returns:
            logits: Classification logits [batch, num_classes]
            representation: The intermediate representation (for analysis)
        """
        # Extract representation using hook
        representation = {}

        def hook_fn(module, module_input, module_output):
            representation["value"] = module_input[0]

        unet = self.np_model.decoder[0]
        handle = unet.final_linear.register_forward_hook(hook_fn)

        try:
            # Run NP model (we don't use its output for classification)
            _ = self.np_model(contexts, xt)
            repr_tensor = representation["value"]
        finally:
            handle.remove()

        # Pool temporal dimension
        pooled = self.pool_temporal(repr_tensor)  # [batch, 512]

        # Apply classifier
        logits = self.classifier(pooled)  # [batch, num_classes]

        return logits, repr_tensor


class LabelledCriteriaTaskLoader(CriteriaTaskLoader):
    """Task loader that yields batches with labels for classification training."""

    def extract_batch_labels(self, batch_ids):
        """Extract binary labels for a batch of tasks."""
        labels = []
        period_labels = self.task_set.period_labels  # type: ignore
        for task_id in batch_ids:
            if task_id in period_labels.index:
                labels.append(period_labels.loc[task_id].astype(int))
            else:
                # This shouldn't happen, but handle gracefully
                raise RuntimeError(f"Task {task_id} not in period_labels")

        return torch.tensor(labels, dtype=torch.long, device=self.device)

    def epoch(self):
        """Generate batches with labels for one epoch."""
        self._batch_index = 0

        def lazy_gen_batch():
            batch = self.generate_batch()
            batch["labels"] = self.extract_batch_labels(batch["ids"])
            return batch

        return (lazy_gen_batch() for _ in range(self.num_batches))


def classification_objective(state, classifier, batch):
    # Forward pass through classifier
    logits, _ = classifier(batch["contexts"], batch["xt"])

    # Compute loss using the classifier's criterion
    loss = classifier.criterion(logits, batch["labels"])
    return state, loss


def predict_classification_probabilities(classifier, batcher):
    classifier.eval()
    task_probs = {}

    with torch.no_grad():
        for batch in batcher.epoch():
            # Forward pass
            logits, _ = classifier(batch["contexts"], batch["xt"])

            # Get probability of positive class
            probs = torch.softmax(logits, dim=1)[:, 1]

            # Store by task ID
            task_ids = batch["ids"]  # type: ignore
            for i, task_id in enumerate(task_ids):
                task_probs[task_id] = probs[i].cpu().item()

    return pd.Series(task_probs)

# This file contains the implementatoin of the Dinov2CLF class. This class is used to define the model architecture for the DINOv2-based model for image classification on ImageNet-Scketch
import torch
import torch.nn as nn
from transformers import AutoModel

class Dinov2CLF(nn.Module):
    def __init__(self, weight_path: str = "facebook/dinov2-giant", frozen_strategy: str = "all", nclasses: int = 500, dropout: float = 0.0, embedding_strategy: str = "cls+seq_emb"):
        super(Dinov2CLF, self).__init__()
        self.embedding_strategy = embedding_strategy
        # Load the pretrained DINOv2 model
        self.feature_extractor = AutoModel.from_pretrained(weight_path)
        
        # Freeze the DINOv2 base model if specified
        if frozen_strategy != "none":
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            if frozen_strategy == "n-1_attention": # freeze all but the last attention layer
                for param in self.feature_extractor.encoder.layers[:-1].parameters():
                    param.requires_grad = True
    
        # Get the size of the DINOv2's output features
        self.hidden_size = self.feature_extractor.config.hidden_size   
        
        if embedding_strategy == "cls":
            self.hidden_size = self.hidden_size
        elif embedding_strategy == "seq_emb":
            self.hidden_size = self.hidden_size
        elif embedding_strategy == "cls+seq_emb":
            self.hidden_size = self.hidden_size * 2
        else:
            raise NotImplementedError("Embedding strategy not implemented")

        # Define a classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, nclasses)
        )

    def forward(self, x):
        # faeture extraction
        x = self.feature_extractor(x)  # Shape: (batch_size, num_tokens, hidden_size)
        # Extract the cls token (first token)
        cls_token = x[0][:, 0, :]  # Shape: (batch_size, hidden_size)
        # Global average pooling across all tokens (excluding class token)
        seq_emb = x[0][:,1:,:].mean(dim=1)  # Shape: (batch_size, hidden_size)
        
        if self.embedding_strategy == "cls":
            pooler_output = cls_token
        elif self.embedding_strategy == "seq_emb":
            pooler_output = seq_emb
        elif self.embedding_strategy == "cls+seq_emb":
            pooler_output = torch.cat([cls_token, seq_emb], dim=1)

        # Pass the pooled representation through the classification head
        return self.classifier(pooler_output)

import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from transformers import AutoImageProcessor

# Transformer implementation from scratch

# one cool thing to add is dropouts
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Shape of pe: [1, max_len, d_model]

        pe = torch.zeros(max_len, d_model) # this will create at most max_len amount of vector with each having a dim of d_model
        position = torch.arange(0, max_len).unsqueeze(1) # [0], [1], [2], ..., [max_len]
        
        index = torch.arange(0, d_model, 2).float() # 0, 2, 4, ..., basically 2*i
        exp_term = index / d_model # 0 / d_model, 2 / d_model, 4 / d_model, ....
        div_term = 10000 ** exp_term # 10000^{0}, 10000^{2/d}, ...
        whole_term = position / div_term # pos / (10000 ^ 2i / d_model)
        
        # 0::2 means from 0th index take 2 steps at a time
        pe[:, 0::2] = torch.sin(whole_term)
        pe[:, 1::2] = torch.cos(whole_term)
        
        pe = pe.unsqueeze(0) # adds a new dimension at index 0, making the shape of pe [1, max_len, d_model]
        # [1, max_len, d_model], since this won't change for any batches, and is precomputed, will have a batch_size of 1
        
        # self.sth parameters are learnable in torch, so will should not use self with it, but we should register it with register_buffer to 
        # allow it to be saved to .pt (when we're saving model's weights and parameters) and also make sure it's loadable to device when
        # we use .to(deivce), so we do sth like this instead of doing self.pe = ....

        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        
        return x + self.pe[:, :x.size(1), :] # since seq_len can be smaller than x, we only take 0 to x.size(1) precomputed pos embeddings
        
        # Shape remains [batch_size, seq_len, d_model]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # in python: x = [thing] * n: Make a list with n references to the same thing object
        # layers = [layer] * 3, layers[0] is layers[1] is layers[2]  # True!
        # All heads will share weights.
        # All gradients will go to the same layer.

        self.w_Q = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(num_heads)])
        self.w_K = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(num_heads)])
        self.w_V = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(num_heads)])

        self.w_O = nn.Linear(d_model, d_model)  # Final projection layer

    def scaled_dot_product_attention(self, Q, K, V):
        # Q, K, V shapes: [batch_size, num_heads, seq_len, d_k]
        
        # Attention(Q, K, V) = softmax(Q @ K^T / √dk) @ V
        
        dk = float(V.shape[-1]) # d_k
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (dk ** 0.5) # K shape: [batch_size, num_heads, d_k, seq_len] 
        weights = torch.softmax(scores, dim=-1) # this will take softmax across the scores
        return torch.matmul(weights, V)
    
        # Shape: [batch_size, num_heads, seq_len, d_k]

    def forward(self, Q, K, V):
        # Q: [batch_size, seq_len, d_model]
        # For each of seq_len tokens in the sequence, we have a d_model-dimensional vector (embedding).

        head_outputs = []
        for head in range(self.num_heads):
            Q_i = self.w_Q[head](Q)  # [batch, seq_len, d_k]
            K_i = self.w_K[head](K)
            V_i = self.w_V[head](V)

            Q_i = Q_i.unsqueeze(1)  # [batch, 1, seq_len, d_k]
            K_i = K_i.unsqueeze(1)
            V_i = V_i.unsqueeze(1)

            attention_i = self.scaled_dot_product_attention(Q_i, K_i, V_i)  # [batch, 1, seq_len, d_k]
            attention_i = attention_i.squeeze(1)  # convert from [batch, 1, seq_len, d_k] to [batch, seq_len, d_k]
            head_outputs.append(attention_i)

        # [batch, seq_len, d_model]
        concat = torch.cat(head_outputs, dim=-1)

        output = self.w_O(concat)  # [batch, seq_len, d_model]
        return output

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.activation = nn.ReLU()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        
        # The FFN:
        # Doesn't care what other tokens say
        # Just looks at its own embedding and transforms it
        # Gives the model more modeling power

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        residual = x
        x = self.multi_head_attention(x, x, x)
        # so the reason to pass only x is cause we're doing self attention, like the input x have multiple patches of image in it
        # and we're trying to see which patches in this input x, we should attend to
        
        x = residual + x
        x = self.norm1(x)

        # Feed-forward
        residual = x
        x = self.feed_forward(x)
        x = residual + x
        x = self.norm2(x)

        return x
        # Shape: [batch_size, seq_len, d_model]

class TransformerEncoder(nn.Module):
    def __init__(self, img_size, patch_size, d_model, num_heads, num_layers, d_ff, num_classes):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = 3 * patch_size * patch_size

        # Learnable linear projection from patch to d_model
        self.patch_embedding = nn.Linear(self.patch_dim, d_model)

        # Positional encoding
        self.pos_embedding = PositionalEncoding(d_model)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])

        # final LayerNorm
        self.norm = nn.LayerNorm(d_model)

        # classification head
        self.fc = nn.Linear(d_model, num_classes)

    def patchify(self, images):
        # images shape: [batch_size, channels, height, width]
        batch_size = images.shape[0]
        # patches shape: [batch_size, num_patches, patch_dim]
        patches = images.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(batch_size, -1, self.patch_dim)
        return patches  # Shape: [batch_size, num_patches, patch_dim]

    def forward(self, x):
        # x: [batch_size, channels, height, width]
        x = self.patchify(x)                      # [B, num_patches, patch_dim]
        x = self.patch_embedding(x)               # [B, num_patches, d_model]
        x = self.pos_embedding(x)                 # add sinusoidal positions

        for layer in self.layers:
            x = layer(x)                          # [B, num_patches, d_model]

        x = self.norm(x)                          # optional final norm

        x = x.mean(dim=1)                         # average all patches
        return self.fc(x)                         # [B, num_classes]

# Data loading and preprocessing
def load_and_preprocess_data():
    # Load the full dataset and split it
    dataset = load_dataset("chriamue/bird-species-dataset", split="train[:5%]")
    train_dataset = dataset.train_test_split(test_size=0.1)  # 10% for validation

    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    
    def preprocess_image(example):
        inputs = image_processor(example["image"], return_tensors="pt")
        return {"pixel_values": inputs.pixel_values.squeeze(0), "label": example["label"]}
    
    train_dataset["train"] = train_dataset["train"].map(preprocess_image, remove_columns=["image"])
    train_dataset["train"].set_format(type="torch", columns=["pixel_values", "label"])
    
    train_dataset["test"] = train_dataset["test"].map(preprocess_image, remove_columns=["image"])
    train_dataset["test"].set_format(type="torch", columns=["pixel_values", "label"])
    
    return train_dataset["train"], train_dataset["test"]

def validate(model, dataloader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    total = 0
    correct = 0
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch["pixel_values"].to(device), batch["label"].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    average_loss = total_loss / len(dataloader)
    print(f"Validation Loss: {average_loss:.3f}, Accuracy: {accuracy:.2f}%")
    
# Training function
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total = 0
    correct = 0
    total_loss = 0
    
    for batch in dataloader:
        inputs, labels = batch["pixel_values"].to(device), batch["label"].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    average_loss = total_loss / len(dataloader)
    print(f"Training loss: {average_loss:.3f}, Accuracy: {accuracy:.2f}%")

# Main function
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameters
    img_size = 224
    patch_size = 16
    d_model = 256
    num_heads = 8
    num_layers = 6
    d_ff = 1024
    num_classes = 525
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.0001
    
    # Load and preprocess data
    train_data, validation_data = load_and_preprocess_data()
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = TransformerEncoder(img_size, patch_size, d_model, num_heads, num_layers, d_ff, num_classes).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training and validation loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train(model, train_loader, criterion, optimizer, device)
        validate(model, validation_loader, criterion, device)

if __name__ == "__main__":
    main()
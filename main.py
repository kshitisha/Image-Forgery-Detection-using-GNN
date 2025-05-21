# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision.transforms as transforms
# import torchvision.models as models
# import numpy as np
# import cv2
# import os
# import matplotlib.pyplot as plt
# import networkx as nx
# from skimage.segmentation import slic
# from skimage.util import img_as_float
# from torch_geometric.data import Data
# from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
# import glob
# from sklearn.model_selection import train_test_split
# from tqdm import tqdm

# # Device setup
# if torch.backends.mps.is_available():
#     device = torch.device("mps")
# elif torch.cuda.is_available():
#     device = torch.device("cuda")
# else:
#     device = torch.device("cpu")
# print(f"Using device: {device}")

# # Load images
# def load_image(forged_path, original_path, num_segments=100):
#     forged = cv2.imread(forged_path)
#     forged = cv2.cvtColor(forged, cv2.COLOR_BGR2RGB)
#     forged = cv2.resize(forged, (256, 256))

#     original = cv2.imread(original_path)
#     original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
#     original = cv2.resize(original, (256, 256))

#     mask = np.abs(forged.astype(np.int16) - original.astype(np.int16)).sum(axis=-1)
#     mask = (mask > 25).astype(np.uint8)

#     img_float = img_as_float(forged)
#     segments = slic(img_float, n_segments=num_segments, compactness=10, sigma=1)
#     return forged, segments, mask

# # Feature extraction
# def extract_features(image, segments, model):
#     transform = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#     ])

#     model.eval()
#     features = []
#     unique_segments = np.unique(segments)

#     with torch.no_grad():
#         for seg_id in unique_segments:
#             mask = (segments == seg_id)
#             patch = image * mask[..., None]
#             patch_tensor = transform(patch).unsqueeze(0).to(device)
#             feature = model(patch_tensor).squeeze().cpu().numpy()
#             features.append(feature)

#     features_array = np.array(features)
#     return torch.tensor(features_array, dtype=torch.float)

# # Create graph
# def create_superpixel_graph(forged_path, original_path, model, num_segments=100):
#     img, segments, mask = load_image(forged_path, original_path, num_segments)
#     node_features = extract_features(img, segments, model).to(device)

#     G = nx.Graph()
#     unique_segments = np.unique(segments)
#     for seg_id in unique_segments:
#         G.add_node(seg_id)

#     for y in range(segments.shape[0] - 1):
#         for x in range(segments.shape[1] - 1):
#             current_segment = segments[y, x]
#             right_segment = segments[y, x + 1]
#             bottom_segment = segments[y + 1, x]
#             if current_segment != right_segment:
#                 G.add_edge(current_segment, right_segment)
#             if current_segment != bottom_segment:
#                 G.add_edge(current_segment, bottom_segment)

#     edge_index = torch.tensor(list(G.edges)).t().contiguous().to(device)
#     labels = np.array([int(np.mean(mask[segments == seg_id]) > 0.5) for seg_id in unique_segments])
#     labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)

#     return Data(x=node_features, edge_index=edge_index, y=labels_tensor), img, segments, mask, G

# # GNN Model
# class ForgeryDetectionGNN(nn.Module):
#     def __init__(self, in_channels=512, hidden_dim=128, out_channels=2):
#         super(ForgeryDetectionGNN, self).__init__()
#         self.conv1 = nn.Linear(in_channels, hidden_dim)
#         self.conv2 = nn.Linear(hidden_dim, out_channels)
#         self.relu = nn.ReLU()

#     def forward(self, x, edge_index):
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.conv2(x)
#         return x

# # Feature extractor
# resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
# resnet.fc = nn.Identity()
# resnet.eval()

# model = ForgeryDetectionGNN().to(device)

# # Dataset
# forged_images = sorted(glob.glob("dataset/Forged/*"))
# original_images = sorted(glob.glob("dataset/Original/*"))
# dataset_paths = list(zip(forged_images, original_images))
# train_paths, test_paths = train_test_split(dataset_paths, test_size=0.2, random_state=42)

# # Training
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# print(" Forgery Detection GNN - Starting Training...")

# for epoch in range(2):
#     print(f" Training Epoch {epoch+1}/2...")
#     total_loss = 0
#     optimizer.zero_grad()

#     for forged_path, original_path in tqdm(train_paths, desc=f" Epoch {epoch+1} Progress"):
#         graph, _, _, _, _ = create_superpixel_graph(forged_path, original_path, resnet)
#         out = model(graph.x, graph.edge_index)
#         loss = criterion(out, graph.y)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()

#     print(f" Epoch {epoch+1} complete | Avg Loss: {total_loss / len(train_paths):.4f}")

# print(" Saving trained model...")
# torch.save(model.state_dict(), "gnn_model.pth")
# print(" Model saved as 'gnn_model.pth'")

# # Evaluation
# print(" Starting model evaluation...")

# model.eval()
# y_true, y_pred = [], []

# os.makedirs("results", exist_ok=True)

# for idx, (forged_path, original_path) in enumerate(tqdm(test_paths, desc="📊 Evaluating")):
#     graph, img, segments, mask, G = create_superpixel_graph(forged_path, original_path, resnet)
#     with torch.no_grad():
#         predictions = model(graph.x, graph.edge_index).argmax(dim=1).cpu().numpy()

#     y_true.extend(graph.y.cpu().numpy())
#     y_pred.extend(predictions)

#     segment_prediction = {seg: pred for seg, pred in zip(np.unique(segments), predictions)}
#     visual = np.zeros_like(mask)
#     for seg_val in np.unique(segments):
#         visual[segments == seg_val] = segment_prediction[seg_val]

#     plt.figure(figsize=(6, 6))
#     plt.imshow(img)
#     plt.imshow(visual, alpha=0.5, cmap='jet')
#     plt.title(f'Predicted Forgery Regions - Sample {idx+1}')
#     plt.axis('off')
#     plt.savefig(f'results/sample_{idx+1}.png')
#     plt.close()

# # Final metrics
# precision = precision_score(y_true, y_pred, average='binary')
# recall = recall_score(y_true, y_pred, average='binary')
# f1 = f1_score(y_true, y_pred, average='binary')
# jaccard = jaccard_score(y_true, y_pred, average='binary')

# print(f" Precision: {precision:.4f}")
# print(f" Recall: {recall:.4f}")
# print(f" F1 Score: {f1:.4f}")
# print(f" IoU: {jaccard:.4f}")
# print(" Evaluation complete! Visualizations saved in 'results/'")



import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import networkx as nx
from skimage.segmentation import slic, mark_boundaries
from skimage.util import img_as_float
from torch_geometric.data import Data
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
import glob
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Device setup
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Load images
def load_image(forged_path, original_path, num_segments=100):
    forged = cv2.imread(forged_path)
    forged = cv2.cvtColor(forged, cv2.COLOR_BGR2RGB)
    forged = cv2.resize(forged, (256, 256))

    original = cv2.imread(original_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    original = cv2.resize(original, (256, 256))

    mask = np.abs(forged.astype(np.int16) - original.astype(np.int16)).sum(axis=-1)
    mask = (mask > 25).astype(np.uint8)

    img_float = img_as_float(forged)
    segments = slic(img_float, n_segments=num_segments, compactness=10, sigma=1)
    return forged, segments, mask

# Feature extraction
def extract_features(image, segments, model, batch_size=128):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    model.eval()
    unique_segments = np.unique(segments)
    patches = []

    for seg_id in unique_segments:
        mask = (segments == seg_id)
        patch = image * mask[..., None]
        patch_tensor = transform(patch)
        patches.append(patch_tensor)

    patches_tensor = torch.stack(patches)  # [N, C, H, W]
    features = []

    with torch.no_grad():
        for i in range(0, len(patches_tensor), batch_size):
            batch = patches_tensor[i:i+batch_size].to(device)
            batch_features = model(batch)
            features.append(batch_features)

    features_tensor = torch.cat(features, dim=0)
    return features_tensor.to(device)

# Create graph
def create_superpixel_graph(forged_path, original_path, model, num_segments=100):
    img, segments, mask = load_image(forged_path, original_path, num_segments)
    node_features = extract_features(img, segments, model).to(device)

    G = nx.Graph()
    unique_segments = np.unique(segments)
    for seg_id in unique_segments:
        G.add_node(seg_id)

    for y in range(segments.shape[0] - 1):
        for x in range(segments.shape[1] - 1):
            current_segment = segments[y, x]
            right_segment = segments[y, x + 1]
            bottom_segment = segments[y + 1, x]
            if current_segment != right_segment:
                G.add_edge(current_segment, right_segment)
            if current_segment != bottom_segment:
                G.add_edge(current_segment, bottom_segment)

    edge_index = torch.tensor(list(G.edges)).t().contiguous().to(device)
    labels = np.array([int(np.mean(mask[segments == seg_id]) > 0.5) for seg_id in unique_segments])
    labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)

    return Data(x=node_features, edge_index=edge_index, y=labels_tensor), img, segments, mask, G

# GNN Model
class ForgeryDetectionGNN(nn.Module):
    def __init__(self, in_channels=512, hidden_dim=128, out_channels=2):
        super(ForgeryDetectionGNN, self).__init__()
        self.conv1 = nn.Linear(in_channels, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x

# Feature extractor
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet.fc = nn.Identity()
resnet = resnet.to(device)
resnet.eval()

model = ForgeryDetectionGNN().to(device)

# Dataset
forged_images = sorted(glob.glob("dataset/Forged/*"))
original_images = sorted(glob.glob("dataset/Original/*"))
dataset_paths = list(zip(forged_images, original_images))
train_paths, test_paths = train_test_split(dataset_paths, test_size=0.2, random_state=42)

# Training
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Forgery Detection GNN - Starting Training...")

for epoch in range(2):
    print(f"Training Epoch {epoch+1}/2...")
    total_loss = 0
    optimizer.zero_grad()

    for forged_path, original_path in tqdm(train_paths, desc=f"🔁 Epoch {epoch+1} Progress"):
        graph, _, _, _, _ = create_superpixel_graph(forged_path, original_path, resnet)
        out = model(graph.x, graph.edge_index)
        loss = criterion(out, graph.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f" Epoch {epoch+1} complete | Avg Loss: {total_loss / len(train_paths):.4f}")

print("Saving trained model...")
torch.save(model.state_dict(), "gnn_model.pth")
print("Model saved as 'gnn_model.pth'")

# Evaluation
print("🧪 Starting model evaluation...")

model.eval()
y_true, y_pred = [], []

os.makedirs("results", exist_ok=True)

for idx, (forged_path, original_path) in enumerate(tqdm(test_paths, desc="📊 Evaluating")):
    graph, img, segments, mask, G = create_superpixel_graph(forged_path, original_path, resnet)
    with torch.no_grad():
        predictions = model(graph.x, graph.edge_index).argmax(dim=1).cpu().numpy()

    y_true.extend(graph.y.cpu().numpy())
    y_pred.extend(predictions)

    segment_prediction = {seg: pred for seg, pred in zip(np.unique(segments), predictions)}
    visual = np.zeros_like(mask)
    for seg_val in np.unique(segments):
        if segment_prediction[seg_val] == 1:
            visual[segments == seg_val] = 255

    img_with_boundaries = mark_boundaries(img, segments)

    plt.figure(figsize=(6, 6))
    plt.imshow(img_with_boundaries)
    plt.imshow(visual, alpha=0.5, cmap='Reds')
    plt.title(f'Predicted Forgery Regions with Boundaries - Sample {idx+1}')
    plt.axis('off')
    plt.savefig(f'results/sample_{idx+1}.png')
    plt.close()

# Final metrics
precision = precision_score(y_true, y_pred, average='binary')
recall = recall_score(y_true, y_pred, average='binary')
f1 = f1_score(y_true, y_pred, average='binary')
jaccard = jaccard_score(y_true, y_pred, average='binary')

print(f"📌 Precision: {precision:.4f}")
print(f"📌 Recall: {recall:.4f}")
print(f"📌 F1 Score: {f1:.4f}")
print(f"📌 IoU: {jaccard:.4f}")
print("✅ Evaluation complete! Visualizations saved in 'results/'")














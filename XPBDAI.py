import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
import joblib
import collections

# 1. Load Data
df = pd.read_csv("BunnyLowObiData.csv")

input_features = [
    'ParticleID', 'Time', 'CollisionParticleID', 'Velocity',
    'Xip', 'Yip', 'Zip', 'Xio', 'Yio', 'Zio'
]
output_features = [
    'Xfp', 'Yfp', 'Zfp', 'Xfo', 'Yfo', 'Zfo'
]

# 2. Time-based split (realistic!)
unique_times = np.sort(df['Time'].unique())
num_train_times = int(0.8 * len(unique_times))
train_times = unique_times[:num_train_times]
test_times = unique_times[num_train_times:]

train_df = df[df['Time'].isin(train_times)]
test_df = df[df['Time'].isin(test_times)]

X_train = train_df[input_features].values.astype(np.float32)
y_train = train_df[output_features].values.astype(np.float32)
X_test = test_df[input_features].values.astype(np.float32)
y_test = test_df[output_features].values.astype(np.float32)

# 3. Normalize (fit only on train, apply to both)
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

# 4. Create DataLoaders
train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)

# 5. Define model
class BunnyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 6)
        )
    def forward(self, x):
        return self.net(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BunnyNet().to(device)

# 6. Loss and optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 7. Train loop
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    avg_loss = total_loss / len(train_ds)
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.5f}")

# 8. Test (Inference) time and MSE
model.eval()
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test).to(device)
    y_test_tensor = torch.tensor(y_test).to(device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()
    y_pred = model(X_test_tensor)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    test_time = time.time() - start
    mse = loss_fn(y_pred, y_test_tensor).item()
    print(f"Test inference time for {len(X_test)} samples: {test_time*1000:.2f} ms")
    print(f"Mean Squared Error: {mse:.6f}")

    # Frame timing: find frame with 89 particles if possible, otherwise largest frame
    frame_counts = collections.Counter(test_df['Time'].values)
    bunny_frames = [t for t, count in frame_counts.items() if count == 89]
    if bunny_frames:
        chosen_time = bunny_frames[0]
        frame_idx = np.where(test_df['Time'].values == chosen_time)[0]
        X_frame = X_test_tensor[frame_idx]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()
        _ = model(X_frame)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        frame_test_time = time.time() - start
        print(f"Time to predict bunny frame (89 particles, Time={chosen_time}): {frame_test_time*1000:.4f} ms")
    else:
        # Use largest available frame
        largest_time, largest_count = max(frame_counts.items(), key=lambda x: x[1])
        frame_idx = np.where(test_df['Time'].values == largest_time)[0]
        X_frame = X_test_tensor[frame_idx]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()
        _ = model(X_frame)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        frame_test_time = time.time() - start
        print(f"No bunny frame (89 particles) found in test set.")
        print(f"Time to predict largest frame ({largest_count} particles, Time={largest_time}): {frame_test_time*1000:.4f} ms")

# 9. Save model and scalers for Unity/Barracuda/ONNX export
torch.save(model.state_dict(), "bunny_deform_model.pt")
joblib.dump(scaler_X, "bunny_scaler_X.pkl")
joblib.dump(scaler_y, "bunny_scaler_y.pkl")
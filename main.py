# main.py
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from data_utils import load_stock_data, prepare_sequences
from model import SimpleDecisionTransformer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")


# 1. Preparar datos
data = load_stock_data("GOOGL")
X, A, R, RTG = prepare_sequences(data)

X_train, X_test, A_train, A_test = train_test_split(X, A, test_size=0.4, shuffle=False)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
A_train = torch.tensor(A_train.squeeze(), dtype=torch.long)
A_test = torch.tensor(A_test, dtype=torch.long).squeeze()

X_train = X_train.to(device)
X_test = X_test.to(device)
A_train = A_train.to(device)
A_test = A_test.to(device)

# 2. Instanciar modelo
model = SimpleDecisionTransformer()
model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 3. Entrenamiento
for epoch in range(20):
    model.train()
    optimizer.zero_grad()
    logits = model(X_train)
    ##print("logits.shape:", logits.shape)
    ##print("A_train.shape:", A_train.shape)
    loss = loss_fn(logits, A_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1} - Loss: {loss.item():.4f}")

# 4. Evaluación
model.eval()
with torch.no_grad():
    preds = model(X_test).argmax(dim=1)
    A_test_cpu = A_test.cpu()
    preds_cpu = preds.cpu()
    print("\nReporte de clasificación (0=Vender, 1=Mantener, 2=Comprar):")
    print(classification_report(A_test_cpu, preds_cpu, digits=4))

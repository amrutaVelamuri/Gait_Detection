import torch
from sklearn.metrics import classification_report, f1_score
import os

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    preds, trues = [], []
    for grf, cop, hand, labels in dataloader:
        grf, cop, hand, labels = grf.to(device), cop.to(device), hand.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(grf, cop, hand)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        preds += torch.argmax(outputs, 1).cpu().tolist()
        trues += labels.cpu().tolist()
    f1 = f1_score(trues, preds, average='macro')
    return f1

def evaluate(model, dataloader, criterion, device, split_name="Val"):
    model.eval()
    preds, trues = [], []
    total_loss = 0
    with torch.no_grad():
        for grf, cop, hand, labels in dataloader:
            grf, cop, hand, labels = grf.to(device), cop.to(device), hand.to(device), labels.to(device)
            outputs = model(grf, cop, hand)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds += torch.argmax(outputs, 1).cpu().tolist()
            trues += labels.cpu().tolist()
    f1 = f1_score(trues, preds, average='macro')
    print(f"\n{split_name} Classification Report:")
    print(classification_report(trues, preds, digits=4))
    print(f"{split_name} F1: {f1:.4f}, Loss: {total_loss / len(dataloader):.4f}\n")
    return f1

def train_model(model, train_loader, val_loader, test_loader, device,
                learning_rate=1e-4, num_epochs=50, patience=10, save_path="saved_models/best_model.pth"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_f1 = 0
    patience_counter = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_f1 = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Train F1: {train_f1:.4f}")
        val_f1 = evaluate(model, val_loader, criterion, device, "Val")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(" Saved Best Model!\n")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(" Early stopping triggered.\n")
                break

    model.load_state_dict(torch.load(save_path))
    print(" Evaluating on Test Set:")
    evaluate(model, test_loader, criterion, device, "Test")

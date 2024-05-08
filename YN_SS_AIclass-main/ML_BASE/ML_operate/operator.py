import torch
from torch import nn
from torch import optim
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 학습 함수
def training(model, train_dataloader, train_dataset, optimizer, criterion):
    print('Training')
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    total = 0
    prog_bar = tqdm(enumerate(train_dataloader), total=int(len(train_dataset)/train_dataloader.batch_size))
    for i, data in prog_bar:
        counter += 1
        data, target = data[0].to(device), data[1].to(device)
        total += target.size(0)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        train_running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == target).sum().item()
        loss.backward()
        optimizer.step()

    train_loss = train_running_loss / counter
    train_accuracy = 100. * train_running_correct / total
    return train_loss, train_accuracy

# 콜백을 적용할 검증 함수
def validate(model, test_dataloader, val_dataset, criterion):
    print('Validating')
    model.eval()
    val_running_loss = 0.0
    val_running_correct = 0
    counter = 0
    total = 0
    prog_bar = tqdm(enumerate(test_dataloader), total=int(len(val_dataset)/test_dataloader.batch_size))
    with torch.no_grad():
        for i, data in prog_bar:
            counter += 1
            data, target = data[0].to(device), data[1].to(device)
            total += target.size(0)
            outputs = model(data)
            loss = criterion(outputs, target)

            val_running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            val_running_correct += (preds == target).sum().item()

        val_loss = val_running_loss / counter
        val_accuracy = 100. * val_running_correct / total
        return val_loss, val_accuracy


### VALIDATION FUNCTION
def validation(model, loader, criterion, device="cpu"):
    model.eval()
    loss = 0
    preds_all = torch.LongTensor()
    labels_all = torch.LongTensor()
    
    with torch.no_grad():
        for batch_x, labels in loader:
            labels_all = torch.cat((labels_all, labels), dim=0)
            batch_x, labels = batch_x.to(device), labels.to(device)
            labels = labels.unsqueeze(1).float()
            
            output = model.forward(batch_x)
            loss += criterion(output,labels).item()
            preds_all = torch.cat((preds_all, output.to("cpu")), dim=0)
    total_loss = loss/len(loader)
    auc_score = roc_auc_score(labels_all, preds_all)
    return total_loss, auc_score

### PREDICTION FUNCTION
def prediction(model, loader, device="cpu"):
    model.to(device)
    model.eval()
    preds_all = torch.LongTensor()
    
    with torch.no_grad():
        for batch_x in loader:
            batch_x = batch_x.to(device)
            
            output = model.forward(batch_x).to("cpu")
            preds_all = torch.cat((preds_all, output), dim=0)
    return preds_all


### TRAINING FUNCTION
def train_model(model, trainloader, validloader, criterion, optimizer, 
                scheduler, epochs=20, device="cpu", print_every=1):
    model.to(device)
    best_auc = 0
    best_epoch = 0
    for e in range(epochs):
        model.train()
        
        for batch_x, labels in trainloader:
            batch_x, labels = batch_x.to(device), labels.to(device)
            labels = labels.unsqueeze(1).float()
            
            # Training 
            optimizer.zero_grad()
            output = model.forward(batch_x)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
        # at the end of each epoch calculate loss and auc score:
        model.eval()
        train_loss, train_auc = validation(model, trainloader, criterion, device)
        valid_loss, valid_auc = validation(model, validloader, criterion, device)
        if valid_auc > best_auc:
            best_auc = valid_auc
            best_epoch = e
            torch.save(model.state_dict(), "best-state.pt")
        if e % print_every == 0:
            to_print = "Epoch: "+str(e+1)+" of "+str(epochs)
            to_print += ".. Train Loss: {:.4f}".format(train_loss)
            to_print += ".. Valid Loss: {:.4f}".format(valid_loss)
            to_print += ".. Valid AUC: {:.3f}".format(valid_auc)
            print(to_print)
    # After Training:
    model.load_state_dict(torch.load("best-state.pt"))
    to_print = "\nTraining completed. Best state dict is loaded.\n"
    to_print += "Best Valid AUC is: {:.4f} after {} epochs".format(best_auc,best_epoch+1)
    print(to_print)
    return
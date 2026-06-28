import torch
import torch.nn as nn
from dataloaders.dataloader import GetLoader
import torch.nn.functional as F
from utils.loss import LossF_BCE
from utils.IoU import GetIoU
from models.model_mask_only import UNet

def main():

    path = './datasets'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE = {device}")

    model = UNet().to(device)

    a = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(a)

    criterion1 = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    TrainLoader1, TestLoader1 = GetLoader(path, 1, 1, 4)

    num_epoch = 60
    for epoch in range(num_epoch):
        model.train()
        train_loss = 0
        TrainLoader, TestLoader = TrainLoader1, TestLoader1

        for imgs, labels, _ in TrainLoader:
            # print(f"labels range: [{labels.min():.3f}, {labels.max():.3f}]")
            # print(f"edges range: [{edges.min():.3f}, {edges.max():.3f}]")
            # print(f"labels unique values: {torch.unique(labels)}")
            # print(f"edges unique values: {torch.unique(edges)}")
            # break
            optimizer.zero_grad()
            imgs, labels = imgs.to(device), labels.to(device)

            mask_pred = model(imgs)
            #前20轮专注mask，后20轮开始看edge
            loss = criterion1(mask_pred, labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()

        avg_trainloss = train_loss / len(TrainLoader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, labels, _ in TestLoader:
                imgs, labels = imgs.to(device), labels.to(device)

                mask_pred = model(imgs)

                loss = criterion1(mask_pred, labels)

                val_loss += loss.item()
        
        avg_testloss = val_loss / len(TestLoader)

        print(f"Epoch {epoch}: Train Loss = {avg_trainloss}, Test Loss = {avg_testloss}")

        # if epoch > 0 and epoch % 10 == 0:
        #     torch.save(model.state_dict(), f'model_epoch_{epoch}.pth')

    torch.save(model.state_dict(), 'model.pth')
    print("保存完了")

if __name__ == '__main__':
    main()
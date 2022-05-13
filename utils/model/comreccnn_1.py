import os
import torch
import torch.nn as nn
import time


class ComCNN(nn.Module):
    def __init__(self):
        super(ComCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.Sigmoid(),
        )

    def forward(self, x):
        output = self.conv_layers(x)
        return output


class RecCNN(nn.Module):
    def __init__(self):
        super(RecCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        output = self.conv_layers(x)
        return output


class nnCceRmoC:
    def __init__(self, net, device):
        super().__init__()
        self.device = device
        self.net = net.to(self.device)
        self.comcnn = ComCNN().to(self.device)
        self.reccnn = RecCNN().to(self.device)
        self.comcnn_path = ""
        self.reccnn_path = ""

    def set_models_path(self, comcnn_path, reccnn_path):
        self.comcnn_path = comcnn_path
        self.reccnn_path = reccnn_path

    def load_models_parameters(self):
        if not os.path.exists(self.comcnn_path):
            raise FileExistsError("comcnn pth file not exist!")

        if not os.path.exists(self.reccnn_path):
            raise FileExistsError("reccnn pth file not exist!")
        self.comcnn.load_state_dict(
            torch.load(self.comcnn_path, map_location=torch.device(self.device))
        )
        self.reccnn.load_state_dict(
            torch.load(self.reccnn_path, map_location=torch.device(self.device))
        )
        print("model parameters loaded!")

    def run(
        self,
        train_loader,
        test_loader,
        epoch=30,
        is_continue=False,
        lr=0.01,
    ):
        if is_continue:
            self.load_models_parameters()
            
        self.net.eval()
        optimizer1 = torch.optim.Adam(
            self.comcnn.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-08,
        )
        optimizer2 = torch.optim.Adam(
            self.reccnn.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-08,
        )
        scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer1, gamma=0.9)
        scheduler2 = torch.optim.lr_scheduler.ExponentialLR(optimizer2, gamma=0.9)

        mse = nn.MSELoss(reduction="mean").to(self.device)
        print("train on {}".format(self.device))

        flag = True
        pre_acc = 0.0
        pre_loss = 0.0
        for e in range(epoch):
            # train
            self.comcnn.train()
            self.reccnn.train()
            start_time = time.time()
            com_l, rec_l = 0, 0
            loss = 0.0
            train_total = 0
            if e != 0 and flag:
                self.load_models_parameters()

            for imgs, lbls in train_loader:
                imgs = imgs.to(self.device)
                lbls = lbls.to(self.device)

                com_out = self.comcnn(imgs)
                # com_loss = mse(com_out, torch.zeros(com_out.shape).to(self.device)).to(self.device)
                com_features = com_out.detach().clone().to(self.device)
                noise = (1**0.5) * torch.randn(com_features.shape).to(self.device)
                # com_features = com_features - noise
                # com_features = torch.round(torch.sigmoid(com_features))
                com_features = torch.round(com_features)
                com_features = com_features - noise
                com_features = torch.sigmoid(com_features)
                com_out.data = com_features.data
                rec_out = self.reccnn(com_out)

                com_loss = mse(com_out, torch.zeros(com_out.shape).to(self.device)).to(self.device)
                rec_loss = mse(rec_out, imgs).to(self.device)

                l = com_loss * 0.0001 + rec_loss / (2 * len(imgs))
                train_total += len(imgs)

                optimizer1.zero_grad()
                optimizer2.zero_grad()
                l.backward()

                optimizer1.step()
                optimizer2.step()

                com_l += com_loss
                rec_l += rec_loss
                loss += com_l + rec_l
            end_time = time.time()
            time_taken = end_time - start_time
            scheduler1.step()
            scheduler2.step()
            com_l /= train_total
            rec_l /= train_total
            loss /= train_total
            print(
                "epoch: {}/{} train loss: {}    Time: {}".format(
                    e + 1, epoch, loss, time_taken
                )
            )
            
            # test
            start_time = time.time()
            succ = 0.0
            test_total = 0.0
            for imgs, lbls in test_loader:
                imgs, lbls = imgs.to(self.device), lbls.to(self.device)
                defend_img = self.defend(imgs)
                
                output = self.net(defend_img)
                pred_indice = output.argmax(1)

                test_total += len(lbls)
                succ += torch.eq(pred_indice, lbls).sum().item()
            end_time = time.time()
            time_taken = end_time - start_time
            print("test accuracy: {}    Time: {}".format(succ / test_total, time_taken))
            if pre_acc < succ / test_total or pre_loss / len(train_loader) > loss / len(
                train_loader
            ):
                flag = False
                torch.save(self.comcnn.state_dict(), self.comcnn_path)
                print("comcnn parameters saved!")
                torch.save(self.reccnn.state_dict(), self.reccnn_path)
                print("reccnn parameters saved!")
                pre_acc = succ / test_total
                pre_loss = loss
            else:
                flag = True
        return (
            com_out,
            rec_out,
        )

    def defend(self, imgs: torch.tensor):
        self.comcnn.eval()
        self.reccnn.eval()
        imgs = imgs.to(self.device)
        com_out = self.comcnn(imgs)
        com_features = com_out.detach().clone()
        noise = (1**0.5) * torch.randn(com_features.shape).to(self.device)
        # com_features = com_features - noise
        # com_features = torch.round(torch.sigmoid(com_features))
        com_features = torch.round(com_features)
        com_features = com_features - noise
        com_features = torch.sigmoid(com_features)
        com_out.data = com_features.data
        rec_out = self.reccnn(com_out)
        return rec_out

import os
import torch
import torch.nn as nn


class resconv(nn.Module):
    def __init__(self, in_channel, out_channel, flag=True):
        super(resconv, self).__init__()
        self.flag = flag
        self.conv = nn.Conv2d(
            in_channel, out_channel, kernel_size=3, bias=True, stride=1, padding=1
        )
        self.conv1x1 = nn.Conv2d(
            in_channel, out_channel, kernel_size=1, bias=False, stride=1
        )
        self.bn = nn.BatchNorm2d(out_channel)
        self.elu = nn.ELU()

    def forward(self, x):
        output = self.conv(x) + self.conv1x1(x)
        if self.flag:
            output = self.bn(output)
            output = self.elu(output)
        return output


class ComCNN(nn.Module):
    def __init__(self):
        super(ComCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ELU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
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
            nn.Conv2d(32, 12, kernel_size=3, padding=1),
            nn.BatchNorm2d(12),
            nn.Sigmoid(),
        )

    def forward(self, x):
        output = self.conv_layers(x)
        return output


class RecCNN(nn.Module):
    def __init__(self):
        super(RecCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, padding=1),
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
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ELU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        output = self.conv_layers(x)
        return output


class ResComCNN(nn.Module):
    def __init__(self):
        super(ResComCNN, self).__init__()
        self.layers = nn.Sequential(
            resconv(3, 16),
            resconv(16, 32),
            resconv(32, 64),
            resconv(64, 128),
            resconv(128, 64),
            resconv(64, 32),
            resconv(32, 12, flag=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        output = self.layers(x)
        return output


class ResRecCNN(nn.Module):
    def __init__(self):
        super(ResRecCNN, self).__init__()
        self.layers = nn.Sequential(
            resconv(12, 32),
            resconv(32, 64),
            resconv(64, 128),
            resconv(128, 64),
            resconv(64, 32),
            resconv(32, 12),
            resconv(12, 3, flag=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        output = self.layers(x)
        return output


class ComRecCNN:
    def __init__(self, net, device, res=False):
        super().__init__()
        self.device = device
        self.net = net.to(self.device)
        if res:
            self.comcnn = ResComCNN().to(self.device)
            self.reccnn = ResRecCNN().to(self.device)
        else:
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

    def train(
        self,
        train_loader,
        test_loader,
        epoch=20,
        is_continue=False,
        lr=0.01,
        alpha=1,
        beta=1,
    ):
        if is_continue:
            self.load_models_parameters()
        self.comcnn.train()
        self.reccnn.train()
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
        scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer1, gamma=0.8)
        scheduler2 = torch.optim.lr_scheduler.ExponentialLR(optimizer2, gamma=0.8)

        mse = nn.MSELoss(reduction="none").to(self.device)
        print("train on {}".format(self.device))
        flag = True
        pre_acc = 0.0
        pre_loss = 0.0
        for e in range(epoch):
            com_l, rec_l = 0, 0
            loss = 0.0
            train_total = 0
            if e != 0 and flag:
                self.load_models_parameters()

            for imgs, lbls in train_loader:
                imgs = imgs.to(self.device)
                lbls = lbls.to(self.device)

                com_out = self.comcnn(imgs)
                com_features = com_out.detach().clone().to(self.device)
                com_features = com_features - (0.1**0.5) * torch.randn(
                    com_features.shape
                ).to(self.device)
                com_features = torch.sigmoid(com_features)
                com_out.data = com_features.data
                rec_out = self.reccnn(com_out)

                zeros = torch.zeros(com_out.shape).to(self.device)

                com_loss = mse(com_out, zeros).mean().to(self.device)
                rec_loss = mse(rec_out, imgs).mean().to(self.device)

                l = alpha * com_loss + beta * rec_loss
                train_total += len(imgs)

                optimizer1.zero_grad()
                optimizer2.zero_grad()
                l.backward()

                optimizer1.step()
                optimizer2.step()

                com_l += alpha * com_loss
                rec_l += beta * rec_loss
                loss += l.item()
            scheduler1.step()
            scheduler2.step()

            print(
                "epoch: {}/{}   com_loss: {}    rec_loss: {}  loss: {}".format(
                    e + 1,
                    epoch,
                    com_l / len(train_loader),
                    rec_l / len(train_loader),
                    loss / len(train_loader),
                )
            )
            succ = 0.0
            test_total = 0.0
            for img, lbl in test_loader:
                img, lbl = img.to(self.device), lbl.to(self.device)
                defend_img = self.defend(img)
                output = self.net(defend_img)
                _, pred_indice = output.max(1)

                test_total += len(lbl)
                succ += (pred_indice == lbl).sum().item()

            print("accuracy: {}".format(succ / test_total))

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

    def defend(self, imgs: torch.tensor):
        imgs = imgs.to(self.device)
        com_out = self.comcnn(imgs)
        com_features = com_out.detach().clone()
        com_features = com_features - (0.1**0.5) * torch.randn(com_features.shape).to(
            self.device
        )
        com_features = torch.round(torch.sigmoid(com_features))
        com_out.data = com_features.data
        rec_out = self.reccnn(com_out)

        return rec_out


if __name__ == "__main__":
    net = ResComCNN()
    print(net)

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms

cifar10_labels = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
imgnet_labels = [
    "tench",
    "brambling",
    "goldfinch",
    "house finch",
    "snowbird",
    "indigo bunting",
    "robin",
    "bulbul",
    "jay",
    "magpie",
]


def showJPEGImage(path: str):
    """
    Show a JPEG image
    :param path:str, path of image
    :return:None
    """
    image = cv2.imread(path)
    cv2.imshow("img", image)
    cv2.waitKey()


def showBatchTensorImage(samples: list, lbls: torch.Tensor) -> None:
    row = len(samples)
    col = samples[0].shape[0]
    count = 0
    _, axes = plt.subplots(row, col, figsize=(8, 5))
    # axes[0][0].set_ylabel("Original               ", rotation=0)
    # axes[1][0].set_ylabel("Preprocessed                         ", rotation=0)
    axes[0][0].set_ylabel("Original               ", rotation=0)
    axes[1][0].set_ylabel("Diff              ", rotation=0)
    # axes[2][0].set_ylabel("Attack              ", rotation=0)
    axes[2][0].set_ylabel("Preprocessed                        ", rotation=0)
    for i in range(row):
        for j in range(col):
            count += 1
            plt.subplot(row, col, count)
            plt.xticks([], [])
            plt.yticks([], [])
            img = np.uint8(samples[i][j] * 255).transpose(1, 2, 0)
            plt.imshow(img)
            axes[i][j].imshow(img)
            axes[0][j].set_title(cifar10_labels[lbls[j].item()])

    plt.tight_layout()
    plt.show()


def showTensorImage(ts: torch.Tensor) -> None:
    """
    Show single tensor image
    :param ts: torch.Tensor, tensor (batch_size, channel, height, width)
    :return: None
    """
    # ts = ts[0]
    ts = ts * 255 / ts.max()
    img = np.uint8(ts).transpose(1, 2, 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()


def JPEG_Defend(ts: torch.Tensor) -> torch.Tensor:
    """
    Show single tensor image
    :param ts: torch.Tensor, tensor (batch_size, channel, height, width)
    :return: torch.Tensor
    """
    defend_img = torch.empty_like(ts)
    toPIL = transforms.ToPILImage()
    for i in range(ts.shape[0]):
        img = toPIL(ts[i]).convert("RGB")
        defend_img[i] = transforms.ToTensor()(img)
    return defend_img


def psnr(img1: np.array, img2: np.array) -> float:
    """
    :param img1:image1
    :param img2:image2
    :return:PSNR of the two image
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    else:
        return 20 * np.log10(255 / np.sqrt(mse))


from PIL import Image


def save_image(save_path, tensor, ori_tensor):
    img = tensor.data.cpu().numpy()
    img = img.transpose(0, 2, 3, 1) * 255.0
    img = np.array(img).astype(np.uint8)
    img = np.concatenate(img, 1)

    ori_img = ori_tensor.data.cpu().numpy()
    ori_img = ori_img.transpose(0, 2, 3, 1) * 255.0
    ori_img = np.array(ori_img).astype(np.uint8)
    ori_img = np.concatenate(ori_img, 1)

    vis = np.concatenate(np.array([ori_img, img]), 0)
    img_pil = Image.fromarray(vis)
    # img_pil = img_pil.resize((w // 16, h // 16))
    img_pil.save(save_path)

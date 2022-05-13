# 毕业设计代码使用说明

本毕业设计使用 Python 编写。

*assets* 文件夹中保存了一些实验结果。

*data* 文件夹中保存了实验用到的数据集 Fashion-MNIST 和 CIFAR-10，代码中调用 torchvision 库自动下载数据集。

*models* 文件夹中分数据集保存了毕设使用的 **ComRecCNN** 和待测网络的参数模型。

*utils* 文件夹中，*attacks* 文件夹保存了实验用多种对抗样本攻击方法实现，*model* 文件夹保存了 **ComRecCNN**、**SimpleNN** 和 **ResNet** 的网络结构代码，*utils* 文件夹保存了一些调用方法，比如数据载入、度量和可视化。

*attack_test.py* 是测试代码。

*train.py* 是 **ComRecCNN** 网络的训练代码。

*victim_model_train.py* 是待测网络的训练代码。

*demo.py* 是一个使用 **FGSM** 的展示小样。

### 环境安装并运行 demo

使用 miniconda 作为 python 包管理器，也可以使用 venv。以 miniconda 为例子演示环境安装。

1. 创建一个虚拟环境

   ```powershell
   conda create -n myenv python=3.10.4
   ```

2. `-y` 安装完成后进入虚拟环境中

   ```powershell
   conda activate myenv
   ```

3. 从 *requirements.txt* 中安装相关的包

   ```powershell
   pip install -r requirements.txt
   ```

4. 进入根目录，运行 *demo.py*

   ```powershell
   python demo.py
   ```

5. 得到 demo 结果

   ```
   FGSM(model_name=ResNet, device=cuda:0, eps=0.007, attack_mode=default, return_type=float)
   攻击下分类正确率0.2927
   防御下分类正确率0.7061,
   ```

### 训练 ComRecCNN

1. 进入根目录，运行 *train.py*

   ```powershell
   python train.py
   ```

2. 训练结果保存在 *model* 文件夹中

from model import PPLCNet
from data import CIFAR10DataModule

import torch
import pytorch_lightning as pl
import os

pl.seed_everything(seed=42)

model = PPLCNet(scale=0.25)
print(f"model is {model.name}")

current_datamodule = CIFAR10DataModule(batch_size=32)
print(f"datamodule is {current_datamodule.name}")

callback1 = pl.callbacks.ModelCheckpoint(dirpath=f'checkpoint/{model.name}', monitor='cross_entropy_val_loss')

version = 1

trainer = pl.Trainer(
    default_root_dir=os.getcwd(), # 默认根目录为当前目录，可以设置为远程连接 hdfs 之类的
    accelerator="auto",
    strategy="ddp2",
    num_nodes=1, #  totalGPUs = gpus * num_nodes
    devices=4,
    auto_select_gpus=True,
    profiler=pl.profiler.SimpleProfiler(),
    max_epochs=5,
    precision=32, # 训练精度：64,32（默认）,16,8，其他精度需要GPU支持
    logger=pl.loggers.TensorBoardLogger(save_dir="logs/", version=version, name=f"{model.name}"),
    log_every_n_steps=50,
    callbacks=[callback1],
    progress_bar_refresh_rate=10, # 每10steps刷新一次进度条
    check_val_every_n_epoch=1, # 每train完1个epoch去validate一次
)

trainer.fit(model=model, datamodule=current_datamodule)

trainer.test(model=model, datamodule=current_datamodule)

# 导出模型,一个是torch script方式，一个是torch trace方式, 一个是onnx格式
model.to_torchscript(file_path=f'./result/{model.name}/version{version}.pt', method='script')
model.to_torchscript(file_path=f'./result/{model.name}/version{version}_trace.pt', method='trace', example_inputs=torch.randn(256, 3, 4, 4))
model.to_onnx(f'./result/{model.name}/version{version}.onnx', input_sample=torch.randn(256, 3, 4, 4), export_params=True)
print(f'suceessfully save model')
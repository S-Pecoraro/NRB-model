import classModel
import torch
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

x = torch.tensor([[1, 8, 3], [4, 5, 6], [7, 8, 9]])
print(x)

writer = SummaryWriter('runs/experiment', flush_secs=1)
t = classModel.plot_confusion_matrix(x.numpy(), ['test','toto','bebe'])
writer.add_figure("confusion matrix on validation set", t, global_step=9)
writer.close()

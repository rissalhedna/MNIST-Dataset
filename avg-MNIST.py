from fastai.vision.all import *
from PIL import Image
import torch

path = untar_data(URLs.MNIST)
#print(path.ls())

training_paths = [(path/'training'/str(i)) for i in range(10)]
testing_paths = [(path/'testing'/str(i)) for i in range(10)]

#print(training_paths)

training_tensors = [torch.stack([tensor(Image.open(l)).float()/255 for l in p.ls()]) for p in training_paths]
testing_tensors = [torch.stack([tensor(Image.open(l)).float()/255 for l in p.ls()]) for p in testing_paths]

print(len(training_tensors), len(testing_tensors))

mean_tensors = [tr.float().mean(0) for tr in training_tensors]

mean_images = [to_image(1- mtr.repeat(3, 1, 1)) for mtr in mean_tensors]

for i in range(10):
    mean_images[i].show()
    
print(testing_tensors[0].shape, mean_tensors[0].shape)

correct = []
total = []

for i in range(10):
  total.append(testing_tensors[i].shape[0])
  preds = torch.Tensor([
          torch.stack(
              [
                F.mse_loss(testing_tensors[i][imgidx], mean_tensors[midx]).sqrt()
                for midx in range(10)
              ]
            ).argmin()
            for imgidx in range(testing_tensors[i].shape[0])
        ])

  correct.append((preds == i).sum())

print(correct, total)


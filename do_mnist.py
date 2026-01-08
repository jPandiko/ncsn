import numpy as np
import tqdm
from losses.dsm import anneal_dsm_score_estimation
from losses.sliced_sm import anneal_sliced_score_estimation_vr
import torch.nn.functional as F
import logging
import torch
import os
import shutil
import tensorboardX
import torch.optim as optim
from torchvision.datasets import MNIST, CIFAR10, SVHN
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from datasets.celeba import CelebA
from models.cond_refinenet_dilated import CondRefineNetDilated
from torchvision.utils import save_image, make_grid
from PIL import Image


# ------ setup parameters here ------------

# training
random_flip = False
image_size = (32,32)
dataset = "MNIST"
batchsize = 128

# optimzers
optimizer_select = "Adam"  # Adam, RMSProp, SGD
learning_rate = 10e-5
weight_decay = 0.000
beta1 = 0.9
amsgrad = False

# file storaging
path = "mnist_run"


# data
channels = 1

class Runner():
  
  def __init__(self):
    # here configs
    1+1
  
  '''
  Method to create the optimizer. We can choose between ADAM, RMSprop, SGD. The setup for the optimizers is 
  stored in the self.config files.
  '''
  def get_optimizer(self, parameters):
      if optimizer_select == 'Adam':
          return optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay,
                              betas=(beta1, 0.999), amsgrad=amsgrad)
      elif optimizer_select == 'RMSProp':
          return optim.RMSprop(parameters, lr=learning_rate, weight_decay=weight_decay)
      elif optimizer_select == 'SGD':
          return optim.SGD(parameters, lr=learning_rate, momentum=0.9)
      else:
          raise NotImplementedError('Optimizer {} not understood.'.format(self.config.optim.optimizer))

  def train(self):
        # 1: transform the datasets into tensors
        if random_flip is False:
            tran_transform = test_transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor()
            ])
        else:
            tran_transform = transforms.Compose([
                transforms.Resize(image_size),
                # fliped horizontally with prob 0.5
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor()
            ])
            test_transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor()
            ])

        # 2: loading the datasets
        dataset = MNIST(os.path.join(self.args.run, 'datasets', 'mnist'), train=True, download=True,
                            transform=tran_transform)
        test_dataset = MNIST(os.path.join(self.args.run, 'datasets', 'mnist_test'), train=False, download=True,
                                 transform=test_transform)


        # 3: setup block for the training
        # wraps dataset so we can iterate in batchesn -> size of batching from configs
        dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True,
                                 num_workers=4, drop_last=True)

        test_iter = iter(test_loader)
        input_dim = image_size ** 2 * channels

        tb_path = os.path.join(self.args.run, 'tensorboard', self.args.doc)
        if os.path.exists(tb_path):
            shutil.rmtree(tb_path)

        # create a data log -> What is logged?
        tb_logger = tensorboardX.SummaryWriter(log_dir=tb_path)
        # Move the score network to device
        score = CondRefineNetDilated(self.config).to(self.config.device)

        score = torch.nn.DataParallel(score)

        optimizer = self.get_optimizer(score.parameters())

        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'))
            score.load_state_dict(states[0])
            optimizer.load_state_dict(states[1])

        step = 0
        
        # create the sigams -> sigmas give the level of noise -> the setup is given from the configurations
        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(self.config.model.sigma_begin), np.log(self.config.model.sigma_end),
                               self.config.model.num_classes))).float().to(self.config.device)

        # 4: training period
        for epoch in range(self.config.training.n_epochs):
            for i, (X, y) in enumerate(dataloader):
                step += 1
                # enable training behavoir of the score-model
                score.train()
                # tranform the discrete data into a number continues set from [0,1]
                X = X.to(self.config.device)
                X = X / 256. * 255. + torch.rand_like(X) / 256.
                # additionally logit tranformation
                if self.config.data.logit_transform:
                    X = self.logit_transform(X)

                # model learns on many noise levels at the same tiem
                labels = torch.randint(0, len(sigmas), (X.shape[0],), device=X.device)
                if self.config.training.algo == 'dsm':
                    loss = anneal_dsm_score_estimation(score, X, labels, sigmas, self.config.training.anneal_power)
                elif self.config.training.algo == 'ssm':
                    loss = anneal_sliced_score_estimation_vr(score, X, labels, sigmas,
                                                             n_particles=self.config.training.n_particles)

                # regular optimizer step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # logging current loss
                tb_logger.add_scalar('loss', loss, global_step=step)
                logging.info("step: {}, loss: {}".format(step, loss.item()))

                if step >= self.config.training.n_iters:
                    return 0

                # every 100 iterations go into evaluation mode
                if step % 100 == 0:
                    score.eval()
                    try:
                        test_X, test_y = next(test_iter)
                    except StopIteration:
                        test_iter = iter(test_loader)
                        test_X, test_y = next(test_iter)

                    test_X = test_X.to(self.config.device)
                    test_X = test_X / 256. * 255. + torch.rand_like(test_X) / 256.
                    if self.config.data.logit_transform:
                        test_X = self.logit_transform(test_X)

                    test_labels = torch.randint(0, len(sigmas), (test_X.shape[0],), device=test_X.device)

                    with torch.no_grad():
                        test_dsm_loss = anneal_dsm_score_estimation(score, test_X, test_labels, sigmas,
                                                                    self.config.training.anneal_power)

                    tb_logger.add_scalar('test_dsm_loss', test_dsm_loss, global_step=step)
                
                # checkpoining -> safe the weights of the network every so and so steps
                if step % self.config.training.snapshot_freq == 0:
                    states = [
                        score.state_dict(),
                        optimizer.state_dict(),
                    ]
                    torch.save(states, os.path.join(self.args.log, 'checkpoint_{}.pth'.format(step)))
                    torch.save(states, os.path.join(self.args.log, 'checkpoint.pth'))

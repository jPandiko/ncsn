import numpy as np
import tqdm
from losses.dsm import dsm_score_estimation
from losses.sliced_sm import anneal_sliced_score_estimation_vr
import torch.nn.functional as F
import logging
import sys
import torch
import os
import shutil
import tensorboardX
import torch.optim as optim
from torchvision.datasets import MNIST, CIFAR10, SVHN
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from datasets.celeba import CelebA
from models.refinenet_dilated_baseline import RefineNetDilated
from torchvision.utils import save_image, make_grid
from argparse import Namespace
from PIL import Image
from argparse import Namespace



# ------ setup parameters here ------------

# training
resume_training = False # !Not Implemented yet ! If you want to resume training, you need to give a checkpoint
random_flip = False
image_size = 32          
dataset_name = "MNIST"
batchsize = 128
n_epochs = 500
n_iters = 201
ngpu = 1
snapshot_freq = 200
algo = "dsm"
anneal_power = 2.0

# optimizers
optimizer_select = "Adam"   # Adam, RMSProp, SGD
learning_rate = 1e-4        # 10e-5 == 1e-4
weight_decay = 0.0
beta1 = 0.9
amsgrad = False

# annealing noise
sigma_begin_para = 0.1
sigma_end_para = 1
num_classes_para = 10
batch_norm_para = False
ngf_para = 64

# file storaging
path = "mnist_run_baseline"

# data
channels = 1
logit_transform = False


# sampling
num_samples = 100
batch_size = 64
n_steps_each = 100
step_lr = 2e-5

# ---------------------------------------------------------


# setup the logger
def setup_logger():
  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,)

def build_config():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return Namespace(
        training=Namespace(
            batch_size=batchsize,
            n_epochs=n_epochs,
            n_iters=n_iters,
            ngpu=ngpu,
            snapshot_freq=snapshot_freq,
            algo=algo,
            anneal_power=anneal_power,
        ),
        data=Namespace(
            dataset=dataset_name,
            image_size=image_size,
            channels=channels,
            logit_transform=logit_transform,
            random_flip=random_flip,
        ),
        model=Namespace(
            sigma_begin=sigma_begin_para, # setup the noise functions
            sigma_end=sigma_end_para,
            num_classes=num_classes_para,
            batch_norm=batch_norm_para,
            ngf=ngf_para,
        ),
        optim=Namespace(
            weight_decay=weight_decay,
            optimizer=optimizer_select,
            lr=learning_rate,
            beta1=beta1,
            amsgrad=amsgrad,
        ),
        device=device,
    )



class Runner():
  
  def __init__(self, config):
    # here configs
    self.config = config
  
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
          raise NotImplementedError('Optimizer {} not understood.'.format(optimizer_select))

    
  def logit_transform(self, image, lam=1e-6):
        image = lam + (1 - 2 * lam) * image
        return torch.log(image) - torch.log1p(-image)

  """
  Method to train the the score network. Parameters are set in the beginning of the file. 
  See that the configs of the run need to be saved manually if the parameters where to be 
  changed between a test run and a training run.
  """
  def train(self):

        # check wether the folder is already existing
        os.makedirs(os.path.join(path, "log"), exist_ok=True)

        if random_flip is False:
            tran_transform = test_transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor()
            ])
        else:
            tran_transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor()
            ])
            test_transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor()
            ])

        dataset = MNIST(os.path.join(path, 'datasets', 'mnist'), train=True, download=True,
                            transform=tran_transform)
        test_dataset = MNIST(os.path.join(path, 'datasets', 'mnist_test'), train=False, download=True,
                                 transform=test_transform)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                                 num_workers=4, drop_last=True)

        print("[+] data loaded", flush=True)
        test_iter = iter(test_loader)
        input_dim = image_size ** 2 * channels

        print("[+] tensorboard storage constructed", flush=True)
        tb_path = os.path.join(path, 'tensorboard')
        if os.path.exists(tb_path):
            shutil.rmtree(tb_path)

        print("[+] build tensor board")
        tb_logger = tensorboardX.SummaryWriter(log_dir=tb_path)

        print("[+] build score network", flush=True)
        score = RefineNetDilated(self.config).to(self.config.device)
        score = torch.nn.DataParallel(score)

        optimizer = self.get_optimizer(score.parameters())

        # WARNING: dont use this, needs to implemented yet
        if resume_training:
            states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'))
            score.load_state_dict(states[0])
            optimizer.load_state_dict(states[1])

        step = 0

        for epoch in range(n_epochs):
            for i, (X, y) in enumerate(dataloader):
                step += 1

                score.train()
                X = X.to(self.config.device)
                X = X / 256. * 255. + torch.rand_like(X) / 256.
                if logit_transform:
                    X = self.logit_transform(X)

                loss = dsm_score_estimation(score, X, sigma=0.01)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tb_logger.add_scalar('loss', loss, global_step=step)
                logging.info("step: {}, loss: {}".format(step, loss.item()))

                if step >= self.config.training.n_iters:
                    return 0

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

                    with torch.no_grad():
                        test_dsm_loss = dsm_score_estimation(score, test_X, sigma=0.01)

                    tb_logger.add_scalar('test_dsm_loss', test_dsm_loss, global_step=step)

                if step % self.config.training.snapshot_freq == 0:
                    states = [
                        score.state_dict(),
                        optimizer.state_dict(),
                    ]
                    torch.save(states, os.path.join(path, "log", 'checkpoint_{}.pth'.format(step)))
                    torch.save(states, os.path.join(path, "log", 'checkpoint.pth'))

  

                
  """
  Used to start the sampling process needed for calculating the scores.
  """
  def start_sampling_images(self):
      # load the model
      states = torch.load(os.path.join(path, "log", "checkpoint.pth"),map_location=self.config.device)

      score = RefineNetDilated(self.config).to(self.config.device)
      score = torch.nn.DataParallel(score)
      score.load_state_dict(states[0])
      score.eval()

      print("[+] model loaded successfully")

      # output directory
      os.makedirs(os.path.join(path, "images"), exist_ok=True)

      device   = self.config.device
      print("[+] directory set up")


      # sampling loop
      global_idx = 0
      pbar = tqdm.tqdm(total=num_samples, desc="Baseline sampling (final samples only)")

      print("[+] start sampling loop")
      while global_idx < num_samples:
        b = min(batch_size, num_samples - global_idx)

        # Initialize from uniform noise
        x = torch.rand(b, channels, image_size, image_size, device=device)

        # Run Langevin dynamics (FINAL samples only)
        x = self.langevin_dynamics_final(x,score,n_steps=n_steps_each,step_lr=step_lr,)

        # Undo logit transform if used during training
        if logit_transform:
            x = torch.sigmoid(x)

        x = x.clamp(0.0, 1.0)

        # 5) Save individual PNGs (metric-ready)
        for j in range(b):
            save_image(
                x[j],
                os.path.join(os.path.join(path, "images"), f"sample_{global_idx:06d}.png")
            )
            global_idx += 1
            pbar.update(1)

        pbar.close()
    
  def langevin_dynamics_final(self,x,scorenet,*,n_steps=2000,step_lr=2e-5,):
      with torch.no_grad():
        for _ in range(n_steps):
            noise = torch.randn_like(x) * torch.sqrt(torch.tensor(2.0 * step_lr, device=x.device))
            grad = scorenet(x)
            x = x + step_lr * grad + noise
      return x
    
  

if __name__ == "__main__":
    setup_logger() # to enable logging
    runner = Runner(build_config())
    runner.start_sampling_images()

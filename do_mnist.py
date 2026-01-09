import numpy as np
import tqdm
from losses.dsm import anneal_dsm_score_estimation
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
from models.cond_refinenet_dilated import CondRefineNetDilated
from torchvision.utils import save_image, make_grid
from argparse import Namespace
from PIL import Image
from argparse import Namespace



# ------ setup parameters here ------------

# training
resume_training = False # !Not Implemented yet! If you want to resume training, you need to give a checkpoint
random_flip = False
image_size = 32          
dataset_name = "MNIST"
batchsize = 128
n_epochs = 5000
n_iters = 2001
ngpu = 1
snapshot_freq = 2000
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
path = "mnist_run"

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
        
        # check wether folder is already existing
        os.makedirs(os.path.join(path, "log"), exist_ok=True)

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
        dataset = MNIST(os.path.join(path, 'datasets', 'mnist'), train=True, download=True,
                            transform=tran_transform)
        test_dataset = MNIST(os.path.join(path, 'datasets', 'mnist_test'), train=False, download=True,
                                 transform=test_transform)


        # 3: setup block for the training
        # wraps dataset so we can iterate in batchesn -> size of batching from configs
        dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True,
                                 num_workers=4, drop_last=True)

        print("[+] data loaded", flush = True)
        test_iter = iter(test_loader)
        input_dim = image_size ** 2 * channels

        print("[+] tensorboard storage constructed", flush = True)
        tb_path = os.path.join(path, 'tensorboard')
        if os.path.exists(tb_path):
            shutil.rmtree(tb_path)
        
        print("[+] build tensor board", flush = True)
        # create a data log 
        tb_logger = tensorboardX.SummaryWriter(log_dir=tb_path)
        
        print("[+] build score network", flush = True)
        # Move the score network to device
        score = CondRefineNetDilated(self.config).to(self.config.device)

        score = torch.nn.DataParallel(score)

        optimizer = self.get_optimizer(score.parameters())

        # WARNING: dont use this, needs to be implemented yet
        if resume_training:
            states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'))
            score.load_state_dict(states[0])
            optimizer.load_state_dict(states[1])

        step = 0
        
        print("[+] build sigmas", flush=True)
        # create the sigams -> sigmas give the level of noise -> the setup is given from the configurations
        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(sigma_begin_para), np.log(sigma_end_para),
                               num_classes_para))).float().to(self.config.device)

        print("[+] start training")
        # 4: training period
        for epoch in range(n_epochs):
            for i, (X, y) in enumerate(dataloader):
                step += 1
                # enable training behavoir of the score-model
                score.train()
                # tranform the discrete data into a number continues set from [0,1]
                X = X.to(self.config.device)
                X = X / 256. * 255. + torch.rand_like(X) / 256.
                # additionally logit tranformation
                if logit_transform:
                    X = self.logit_transform(X)

                # model learns on many noise levels at the same tiem
                labels = torch.randint(0, len(sigmas), (X.shape[0],), device=X.device)
                if algo == 'dsm':
                    loss = anneal_dsm_score_estimation(score, X, labels, sigmas, anneal_power)
                #elif self.config.training.algo == 'ssm':
                #    loss = anneal_sliced_score_estimation_vr(score, X, labels, sigmas,
                #                                            n_particles=self.config.training.n_particles)

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
                        test_dsm_loss = anneal_dsm_score_estimation(score, test_X, test_labels, sigmas,anneal_power)

                    tb_logger.add_scalar('test_dsm_loss', test_dsm_loss, global_step=step)
                
                # checkpoining -> safe the weights of the network every so and so steps
                if step % snapshot_freq == 0:
                    print("[+] save ceckpoint")
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
        states = torch.load(os.path.join(path, "log", 'checkpoint.pth'), map_location=self.config.device)
        score = CondRefineNetDilated(self.config).to(self.config.device)
        score = torch.nn.DataParallel(score)
        score.load_state_dict(states[0])

        print("[+] sampling setup prepared successfuly")

        self.sample_images(score,num_samples=num_samples,batch_size=batch_size,n_steps_each = n_steps_each,step_lr=step_lr,)

    
  """
  Generate samples using annealed Langevin dynamics and save them as individual PNGs.
  Suitable for FID / Inception Score evaluation.
  """
  def sample_images(self,scorenet,*,num_samples: int,batch_size: int = 64,n_steps_each: int = 100,step_lr: float = 2e-5,):
      
      scorenet.eval()
      os.makedirs(os.path.join(path, "images"), exist_ok=True)

      # --- build sigma schedule (torch, on device) ---
      sigmas = torch.tensor(np.exp(np.linspace(
            np.log(sigma_begin_para),
            np.log(sigma_end_para),
            num_classes_para,
        )),
        dtype=torch.float32,
        device=self.config.device)

      size = image_size
      device = self.config.device

      global_idx = 0

      with torch.no_grad():
          while global_idx < num_samples:
            print("[+] current idx :", global_idx)
            b = min(batch_size, num_samples - global_idx)

            # --- initialize from uniform noise ---
            x = torch.rand(b, channels, size, size, device=device)

            print("[+] init randomized")

            # --- annealed Langevin dynamics ---
            for c, sigma in enumerate(sigmas):
                labels = torch.full((b,), c, device=device, dtype=torch.long)
                step_size = step_lr * (sigma / sigmas[-1]) ** 2

                for _ in range(n_steps_each):
                    noise = torch.randn_like(x) * torch.sqrt(step_size * 2)
                    grad = scorenet(x, labels)
                    x = x + step_size * grad + noise

            # --- invert logit transform if used during training ---
            if self.config.data.logit_transform:
                x = torch.sigmoid(x)

            x = x.clamp(0.0, 1.0)

            # --- save individual images ---
            for j in range(b):
                save_image(x[j],os.path.join(path,"images", f"sample_{global_idx:06d}.png"))
                global_idx += 1


if __name__ == "__main__":
    setup_logger() # to enable logging
    runner = Runner(build_config())
    runner.train()
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Discrimimator, Generator, initialize_weights
from utils import gradient_penalty


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 5e-5
# LEARNING_RATE = 1e-4  for WGAN-GP
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 1
NOISE_DIM = 100
NUM_EPOCH = 5
FEATURES_DISC = 64
FEATURES_GEN = 64
CRITIC_ITERATIONS = 5
WEIGHT_CLIP = 0.01 # no WEIGHT_CLIP IN WGAN-GP
# LAMBDA_GP = 10 -- in WGAN-GP

transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)])
    ]
)

dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms, download=True)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
gen = Generator(NOISE_DIM, CHANNELS_IMG, CHANNELS_IMG).to(device)
critic = Discrimimator(CHANNELS_IMG, FEATURES_DISC).to(device)
initialize_weights(gen)
initialize_weights(critic)

opt_gen = optim.RMSprop(gen.parameters(), lr=LEARNING_RATE)
# opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))  in WGAN-GP
opt_disc = optim.RMSprop(critic.parameters(), lr=LEARNING_RATE)
# opt_disc = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
criterion = nn.BCELoss()

fixed_noise = torch.randn(32, NOISE_DIM, 1, 1).to(device)
writer_fake = SummaryWriter(f"logs/fake")
writer_real = SummaryWriter(f"logs/real")
step = 0

gen.train()
critic.train()

for epoch in range(NUM_EPOCH):
    # target labels not needed! unsupervised
    for batch_idx, (real, _) in enumerate(loader):
        data = data.to(device)
        cur_batch_size = data.shape[0]

        #train critic: max E[critic(real)] - E[critic(fake)]
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn(cur_batch_size, NOISE_DIM, 1, 1).to(device)
            fake = gen(noise)
            critic_real = critic(data).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            # gp = gradient_penalty(critic, real, fake, device)
            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake))
            # loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake) + LAMBDA_GP * gp)
            critic.zero_grad()
            loss_critic.backward(retian_graph=True)
            opt_disc.step()

            # clip critic weight between -0.01, 0.01, 梯度截断，将梯度约束在[-grad_clip, grad_clip]区间中
            for p in critic.parameters(): # no this step in WGAN-GP
                p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

        # train generator: max E[critic(gen_fake)] -- min -E[critic(gen_fake)]
        gen_fake = critic(fake).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # print losses occasionally and print to tensorborad
        if batch_idx % 100 == 0 and batch_idx > 0:
            gen.eval()
            critic.eval()
            print(f"Epoch[{epoch} / {NUM_EPOCH}] Batch {batch_idx} / {len(loader)} Loss D: {loss_critic:.4f} Loss G: {loss_gen:.4f}")

            with torch.no_grad():
                fake = gen(noise)
                # take out (up to) 32 eamaples
                img_grid_real = torchvision.utils.make_grid(
                    data[: 32], normalize=True
                )
                img_grid_fake = torchvision.utils.make_grid(
                    fake[: 32], normalize=True
                )
                writer_real.add_image("real", img_grid_real, global_step=step)
                writer_fake.add_image("fake", img_grid_fake, global_step=step)
                step += 1
                gen.train()
                critic.train()

# Importing the libraries
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel as parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

#! JUST A HEADS UP
"""
This section uses alot of references to a well known tensorflow notebook on its interpretations on GANS
Here is the link:- https://www.tensorflow.org/tutorials/generative/dcgan 

"""

# Setting the hyperparamaters
batchSize = 30
imageSize = 256

# Getting the dataset
dataset_dir = r"C:\Users\Selvaseetha\Kaggle Competitions\(2) I’m Something of a Painter Myself\monet_jpg"
dataloader = torch.utils.data.DataLoader(
    dataset_dir, batch_size=batchSize, shuffle=True, num_workers=True)
#                                                    Number of threads

# Function that will initialize the weights


def weights_init(m):
    class_name = m.__class__.__name__
    if class_name.find("Conv"):
        m.weight.data.normal_(0.0, 0.2)
    elif class_name.find("BatchNorm") != 1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


#! PART 1 - CREATING

# Defining the generator

class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=100,
                               out_channels=512,
                               kernel_size=4,
                               stride=1,
                               padding=0,  # ! In the tensorflow doccumentation padding was used
                               bias=False),
            nn.BatchNorm2d(num_features=512),
            # ! In the tensorflow doccumentation and in the paper, Leaky ReLU was used
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=512,
                               out_channels=256,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=256,
                               out_channels=128,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=128,
                               out_channels=64,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=64,
                               out_channels=3,  # It's 3 as the image is coloures, it must have 3 dimesnions
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.Tanh()
        )

    # Function the forward propagates an input
    def forward(self, input_noise):
        output = self.main(input_noise)
        return output


# Creating the generator and initializing its weights
generator = G()
generator.apply(weights_init)

# Defining the Discriminator


class D(nn.Module):
    def __init__(self):
        super(self, D).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=3,  # ! The 3 channels generated from the generator
                      out_channels=64,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.LeakyReLU(negative_slope=0.2,  # Controls angle of the negative slope
                         inplace=True),
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
            # This feature was non-existant in the tensorflow docs
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(negative_slope=0.2,
                         inplace=True),
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(negative_slope=0.2,
                         inplace=True),
            nn.Conv2d(in_channels=256,
                      out_channels=512,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(negative_slope=0.2,
                         inplace=True),
            nn.Conv2d(in_channels=512,
                      out_channels=1,
                      kernel_size=4,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.Sigmoid()  # As output values must be between 0 & 1

        )

    # Propagation of the discriminator
    def forward(self, generated_img):
        output = self.main(generated_img)
        # ! This flattens the values of the CNN from 2D to dimension 1 using a values of -1
        return output.view(-1)


# Creating and initializing the discriminator
discriminator = D()
discriminator.apply(weights_init)

#! PART 2 - TRAINING
loss = nn.BCELoss()  # Both values have common losses like in the tensorflow doccumentation
optimD = optim.Adam(params=discriminator.parameters(),
                    lr=0.0002,
                    betas=(0.5, 0.999)
                    )
optimG = optim.Adam(params=generator.parameters(),
                    lr=0.0002,
                    betas=(0.5, 0.999)
                    )

#! Going through the epochs
for epoch in range(25):
    for i, data in enumerate(dataloader, 0):
        # Updating the weights of the discriminator
        discriminator.zero_grad()
        # Getting the error of real images
        real, _ = data
        real_input = Variable(real)
        real_output = discriminator(real_input)
        target = Variable(torch.ones(real_input.size()[0]))
        disc_error = loss(real_output, target)
        # Training using fake images from the generator
        noise = Variable(torch.randn((real_input.size()[0], 100, 1, 1)))
        fake_output = generator(noise)
        target = Variable(torch.zeros(real_input.size()[0]))
        output = discriminator(fake_output.detatch())
        disc_error_fake = loss(output, target)
        # Backpropagating the total error
        final_error_disc = disc_error + disc_error_fake
        final_error_disc.backward()
        optimD.step()  # Aplies the optimizer to the network
        # Updating the weights for the generator
        generator.zero_grad()
        target = Variable(torch.ones(real_input.size()[0]))
        output = discriminator(fake_output)
        error_gen = loss(output, target)
        error_gen.backward()
        optimG.step()  # Aplies the optimizer to the network
        # Printing all the metrics of evaluation
        print(
            f"[{str(epoch)}/25][{str(i)}/{len(dataloader)}], {str(disc_error.data[0])}, {error_gen.data[0]}")
        # Saving the images
        fake = generator(noise)
        vutils.save_image(
            fake.data, f"Image of epoch {epoch}.png", normailze=True)

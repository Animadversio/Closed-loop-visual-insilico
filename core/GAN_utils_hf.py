""" pre-trained GAN models for feature visualization
Examples:
    # Load a state dict
    G = upconvGAN("fc6")
    # Load a state dict from a pretrained model
    G = upconvGAN("pool5")
    G.G.load_state_dict(torch.hub.load_state_dict_from_url(r"https://drive.google.com/uc?export=download&id=1vB_tOoXL064v9D6AKwl0gTs1a7jo68y7",progress=True))
"""
import os
from os.path import join
from sys import platform
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

model_urls = {
    "caffenet": "https://huggingface.co/binxu/DeePSim_DosovitskiyBrox2016/resolve/main/caffenet.pt",
    "norm1": "https://huggingface.co/binxu/DeePSim_DosovitskiyBrox2016/resolve/main/upconvGAN_norm1.pt",
    "norm2": "https://huggingface.co/binxu/DeePSim_DosovitskiyBrox2016/resolve/main/upconvGAN_norm2.pt",
    "conv3": "https://huggingface.co/binxu/DeePSim_DosovitskiyBrox2016/resolve/main/upconvGAN_conv3.pt",
    "conv4": "https://huggingface.co/binxu/DeePSim_DosovitskiyBrox2016/resolve/main/upconvGAN_conv4.pt",
    "pool5": "https://huggingface.co/binxu/DeePSim_DosovitskiyBrox2016/resolve/main/upconvGAN_pool5.pt",
    "fc6": "https://huggingface.co/binxu/DeePSim_DosovitskiyBrox2016/resolve/main/upconvGAN_fc6.pt",
    "fc6_eucl": "https://huggingface.co/binxu/DeePSim_DosovitskiyBrox2016/resolve/main/upconvGAN_fc6_eucl.pt",
    "fc7": "https://huggingface.co/binxu/DeePSim_DosovitskiyBrox2016/resolve/main/upconvGAN_fc7.pt",
    "fc8": "https://huggingface.co/binxu/DeePSim_DosovitskiyBrox2016/resolve/main/upconvGAN_fc8.pt",
}

def download_file(url, local_path):
    import requests
    from tqdm.auto import tqdm
    response = requests.get(url, stream=True)
    file_size = int(response.headers.get('content-length', 0))
    chunk_size = 1024
    progress_bar = tqdm(total=file_size, unit='iB', unit_scale=True)

    with open(local_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=chunk_size):
            progress_bar.update(len(chunk))
            file.write(chunk)

    progress_bar.close()
    if file_size != 0 and progress_bar.n != file_size:
        print("ERROR, something went wrong.")


def load_statedict_from_online(name="fc6"):
    import requests
    from requests.exceptions import HTTPError
    torchhome = torch.hub.get_dir()
    ckpthome = join(torchhome, "checkpoints")
    os.makedirs(ckpthome, exist_ok=True)
    filepath = join(ckpthome, "upconvGAN_%s.pt"%name)
    if not os.path.exists(filepath):
        print(f"downloading weights of `upconvGAN_{name}.pt` to {filepath}")
        download_file(model_urls[name], filepath)

    SD = torch.load(filepath)
    return SD


class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class ScalarMultiply(nn.Module):
    def __init__(self, scalar):
        super(ScalarMultiply, self).__init__()
        self.scalar = scalar

    def forward(self, x):
        return self.scalar * x


RGB_mean = torch.tensor([123.0, 117.0, 104.0])
RGB_mean = torch.reshape(RGB_mean, (1, 3, 1, 1))
class upconvGAN(nn.Module):
    def __init__(self, name="fc6", pretrained=True):
        super(upconvGAN, self).__init__()
        self.name = name
        if name == "fc6" or name == "fc7" or name == "fc6_eucl":
            self.G = nn.Sequential(OrderedDict([
        ('defc7', nn.Linear(in_features=4096, out_features=4096, bias=True)),
        ('relu_defc7', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('defc6', nn.Linear(in_features=4096, out_features=4096, bias=True)),
        ('relu_defc6', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('defc5', nn.Linear(in_features=4096, out_features=4096, bias=True)),
        ('relu_defc5', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('reshape', View((-1, 256, 4, 4))),
        ('deconv5', nn.ConvTranspose2d(256, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv5', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('conv5_1', nn.ConvTranspose2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('relu_conv5_1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('deconv4', nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv4', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('conv4_1', nn.ConvTranspose2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('relu_conv4_1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('deconv3', nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv3', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('conv3_1', nn.ConvTranspose2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('relu_conv3_1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('deconv2', nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('deconv1', nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('deconv0', nn.ConvTranspose2d(32, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
            ]))
            self.codelen = self.G[0].in_features
            self.latent_shape = (4096,)
        elif name == "fc8":
            self.G = nn.Sequential(OrderedDict([
        ("defc7", nn.Linear(in_features=1000, out_features=4096, bias=True)),
        ("relu_defc7", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ("defc6", nn.Linear(in_features=4096, out_features=4096, bias=True)),
        ("relu_defc6", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ("defc5", nn.Linear(in_features=4096, out_features=4096, bias=True)),
        ("relu_defc5", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ("reshape", View((-1, 256, 4, 4))),
        ("deconv5", nn.ConvTranspose2d(256, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ("relu_deconv5", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ("conv5_1", nn.ConvTranspose2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ("relu_conv5_1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ("deconv4", nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ("relu_deconv4", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ("conv4_1", nn.ConvTranspose2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ("relu_conv4_1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ("deconv3", nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ("relu_deconv3", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ("conv3_1", nn.ConvTranspose2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ("relu_conv3_1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ("deconv2", nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ("relu_deconv2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ("deconv1", nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ("relu_deconv1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ("deconv0", nn.ConvTranspose2d(32, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ]))
            self.codelen = self.G[0].in_features
            self.latent_shape = (1000,)
        elif name == "pool5":
            self.G = nn.Sequential(OrderedDict([
        ('Rconv6', nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('Rrelu6', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('Rconv7', nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('Rrelu7', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('Rconv8', nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1))),
        ('Rrelu8', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('deconv5', nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv5', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('conv5_1', nn.ConvTranspose2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('relu_conv5_1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('deconv4', nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv4', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('conv4_1', nn.ConvTranspose2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('relu_conv4_1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('deconv3', nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv3', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('conv3_1', nn.ConvTranspose2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('relu_conv3_1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('deconv2', nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('deconv1', nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('deconv0', nn.ConvTranspose2d(32, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))), ]))
            self.codelen = self.G[0].in_channels
            self.latent_shape = (256, 6, 6)
        elif name == "conv4":
            self.G = nn.Sequential(OrderedDict([
                ('Rconv6', nn.Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1))),
                ('Rrelu6', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                ('Rconv7', nn.Conv2d(384, 512, kernel_size=(3, 3), stride=(1, 1))),
                ('Rrelu7', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                ('Rconv8', nn.Conv2d(512, 512, kernel_size=(2, 2), stride=(1, 1))),
                ('Rrelu8', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                ('deconv5', nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
                ('relu_deconv5', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                ('conv5_1', nn.ConvTranspose2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                ('relu_conv5_1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                ('deconv4', nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
                ('relu_deconv4', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                ('conv4_1', nn.ConvTranspose2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                ('relu_conv4_1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                ('deconv3', nn.ConvTranspose2d(128, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
                ('relu_deconv3', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                ('conv3_1', nn.ConvTranspose2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                ('relu_conv3_1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                ('deconv2', nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
                ('relu_deconv2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                ('conv2_1', nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                ('relu_conv2_1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                ('deconv1', nn.ConvTranspose2d(32, 16, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
                ('relu_deconv1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                ('conv1_1', nn.Conv2d(16, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                ('tanh', nn.Tanh()),
                ("scaling", ScalarMultiply(255.0)),
            ]))
            self.codelen = self.G[0].in_channels
            self.latent_shape = (384, 13, 13)
        elif name == "conv3":
            self.G = nn.Sequential(OrderedDict([
                ('Rconv6', nn.Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1))),
                ('Rrelu6', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                ('Rconv7', nn.Conv2d(384, 512, kernel_size=(3, 3), stride=(1, 1))),
                ('Rrelu7', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                ('Rconv8', nn.Conv2d(512, 512, kernel_size=(2, 2), stride=(1, 1))),
                ('Rrelu8', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                ('deconv5', nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
                ('relu_deconv5', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                ('conv5_1', nn.ConvTranspose2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                ('relu_conv5_1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                ('deconv4', nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
                ('relu_deconv4', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                ('conv4_1', nn.ConvTranspose2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                ('relu_conv4_1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                ('deconv3', nn.ConvTranspose2d(128, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
                ('relu_deconv3', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                ('conv3_1', nn.ConvTranspose2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                ('relu_conv3_1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                ('deconv2', nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
                ('relu_deconv2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                ('conv2_1', nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                ('relu_conv2_1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                ('deconv1', nn.ConvTranspose2d(32, 16, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
                ('relu_deconv1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                ('conv1_1', nn.Conv2d(16, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                ('tanh', nn.Tanh()),
                ("scaling", ScalarMultiply(255.0)),
            ]))
            self.codelen = self.G[0].in_channels
            self.latent_shape = (384, 13, 13)
        elif name == "norm2":
            self.G = nn.Sequential(OrderedDict([
                ('Rconv6', nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2))),
                ('Rrelu6', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                ('Rconv7', nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                ('Rrelu7', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                ('Rconv8', nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                ('Rrelu8', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                ('deconv4', nn.ConvTranspose2d(128, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
                ('relu_deconv4', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                ('conv4_1', nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                ('relu_conv4_1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                ('deconv3', nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
                ('relu_deconv3', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                ('conv3_1', nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                ('relu_conv3_1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                ('deconv2', nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
                ('relu_deconv2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                ('conv2_1', nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                ('relu_conv2_1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                ('deconv1', nn.ConvTranspose2d(32, 16, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
                ('conv1_1', nn.Conv2d(16, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                ('tanh', nn.Tanh()),
                ("scaling", ScalarMultiply(255.0)),
            ]))
            self.codelen = self.G[0].in_channels
            self.latent_shape = (256, 13, 13)
        elif name == "norm1":
            self.G = nn.Sequential(OrderedDict([
                    ('Rconv6', nn.Conv2d(96, 128, kernel_size=(3, 3), stride=(2, 2), padding=(2, 2))),
                    ('Rrelu6', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                    ('Rconv7', nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                    ('Rrelu7', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                    ('Rconv8', nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                    ('Rrelu8', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                    ('deconv4', nn.ConvTranspose2d(128, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
                    ('relu_deconv4', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                    ('conv4_1', nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                    ('relu_conv4_1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                    ('deconv3', nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
                    ('relu_deconv3', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                    ('conv3_1', nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                    ('relu_conv3_1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                    ('deconv2', nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
                    ('relu_deconv2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                    ('conv2_1', nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                    ('relu_conv2_1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                    ('deconv1', nn.ConvTranspose2d(32, 16, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
                    ('conv1_1', nn.Conv2d(16, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                    ('tanh', nn.Tanh()),
                    ("scaling", ScalarMultiply(255.0)),
                ]))
            self.codelen = self.G[0].in_channels
            self.latent_shape = (96, 27, 27)
        # load pre-trained weight from online or local folders
        if pretrained:
            SD = load_statedict_from_online(name)
            SDnew = OrderedDict()
            for name, W in SD.items():  # discard this inconsistency
                name = name.replace(".1.", ".")
                SDnew[name] = W
            self.G.load_state_dict(SDnew)

    def sample_vector(self, sampn=1, device="cuda", noise_std=1.0):
        return torch.randn(sampn, self.codelen, device=device) * noise_std

    def forward(self, x):
        return self.G(x)[:, [2, 1, 0], :, :]

    def visualize(self, x, scale=1.0):
        raw = self.G(x)[:, [2, 1, 0], :, :]
        return torch.clamp(raw + RGB_mean.to(raw.device), 0, 255.0) / 255.0 * scale

    def visualize_batch(self, x_arr, scale=1.0, B=42, ):
        coden = x_arr.shape[0]
        img_all = []
        csr = 0  # if really want efficiency, we should use minibatch processing.
        with torch.no_grad():
            while csr < coden:
                csr_end = min(csr + B, coden)
                imgs = self.visualize(x_arr[csr:csr_end, :].cuda(), scale).cpu()
                img_all.append(imgs)
                csr = csr_end
        img_all = torch.cat(img_all, dim=0)
        return img_all

    def render(self, x, scale=1.0, B=42):  # add batch processing to avoid memory over flow for batch too large
        coden = x.shape[0]
        img_all = []
        csr = 0  # if really want efficiency, we should use minibatch processing.
        while csr < coden:
            csr_end = min(csr + B, coden)
            with torch.no_grad():
                imgs = self.visualize(torch.from_numpy(x[csr:csr_end, :]).float().cuda(), scale).permute(2,3,1,0).cpu().numpy()
            img_all.extend([imgs[:, :, :, imgi] for imgi in range(imgs.shape[3])])
            csr = csr_end
        return img_all

    def visualize_batch_np(self, codes_all_arr, scale=1.0, B=42, verbose=False):
        coden = codes_all_arr.shape[0]
        img_all = None
        csr = 0  # if really want efficiency, we should use minibatch processing.
        with torch.no_grad():
            while csr < coden:
                csr_end = min(csr + B, coden)
                imgs = self.visualize(torch.from_numpy(codes_all_arr[csr:csr_end, :]).float().cuda(), scale).cpu()
                img_all = imgs if img_all is None else torch.cat((img_all, imgs), dim=0)
                csr = csr_end
        return img_all


class Caffenet(nn.Module):
    def __init__(self, pretrained=True, ):
        super(Caffenet, self).__init__()
        self.net = nn.Sequential(OrderedDict([
                # Layer 1
                ('conv1', nn.Conv2d(3, 96, kernel_size=(11, 11), stride=(4, 4))),
                ('relu1', nn.ReLU(inplace=True)),
                ('pool1', nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)),
                ('norm1', nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1.0)),
                # Layer 2
                ('conv2', nn.Conv2d(96, 256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=2)),
                ('relu2', nn.ReLU(inplace=True)),
                ('pool2', nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)),
                ('norm2', nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1.0)),
                # Layer 3
                ('conv3', nn.Conv2d(256, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                ('relu3', nn.ReLU(inplace=True)),
                # Layer 4
                ('conv4', nn.Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=2)),
                ('relu4', nn.ReLU(inplace=True)),
                # Layer 5
                ('conv5', nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=2)),
                ('relu5', nn.ReLU(inplace=True)),
                ('pool5', nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)),
                # Fully connected layers
                ('flatten', nn.Flatten()),
                ('fc6', nn.Linear(9216, 4096)),
                ('relu6', nn.ReLU(inplace=True)),
                ('fc7', nn.Linear(4096, 4096)),
                ('relu7', nn.ReLU(inplace=True)),
                ('fc8', nn.Linear(4096, 1000))
            ]))
        if pretrained:
            SD = load_statedict_from_online("caffenet")
            SDnew = OrderedDict()
            for name, W in SD.items():  # discard this inconsistency
                name = name.replace(".1.", ".")
                SDnew[name] = W
            self.net.load_state_dict(SDnew)
    
    def forward(self, x, preproc=False, scale=1.0):
        if preproc:
            x = x.float() / scale * 255.0
            x = x - RGB_mean.to(x.device)
            x = x[:, [2, 1, 0], :, :]
        return self.net(x)


import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal


class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(VAE, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size * 2)  # 输出均值和方差
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )

    def encode(self, x):
        # 计算均值和方差
        h = self.encoder(x)
        mu, log_var = h[:, :self.latent_size], h[:, self.latent_size:]
        return mu, log_var

    def reparameterize(self, mu, log_var):
        # 从分布中采样一个随机向量z
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        # 解码器
        return self.decoder(z)

    def forward(self, x):
        # 编码器-重参数化-解码器
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        return x_hat, mu, log_var

    def loss_function(self, x, x_hat, mu, log_var):
        # VAE的损失函数，由重建误差和KL散度组成
        recon_loss = nn.functional.binary_cross_entropy_with_logits(x_hat, x, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + kl_div


# 使用示例
input_size = 784
hidden_size = 256
latent_size = 16
vae = VAE(input_size, hidden_size, latent_size)
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

for epoch in range(10):
    for x, _ in train_loader:
        x = x.view(-1, input_size)
        x_hat, mu, log_var = vae(x)
        loss = vae.loss_function(x, x_hat, mu, log_var)
        optimizer

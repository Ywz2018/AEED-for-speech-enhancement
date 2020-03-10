import torch

SSNR_FRAME = 512

def calc_ssnr(deg, clean):
    bt = deg.size(0)  # BATCHSIZE
    T = deg.size(1)  # TIMESTAMP
    # Reshape to facilitate transform
    surplus = T % SSNR_FRAME
    deg = deg[:, :-surplus]
    clean = clean[:, :-surplus]

    deg = deg / torch.abs(torch.mean(deg, dim=1, keepdim=True))
    clean = clean / torch.abs(torch.mean(clean, dim=1, keepdim=True))

    deg = deg.view(bt, -1, SSNR_FRAME)
    clean = clean.view(bt, -1, SSNR_FRAME)
    noise = clean - deg
    clean_energy = torch.sum(clean ** 2, dim=2) + 1e-6
    noise_energy = torch.sum(noise ** 2, dim=2) + 1e-6
    ssnr = 10 * torch.log(clean_energy / noise_energy) / np.log(10)
    ssnr = torch.clamp(ssnr, min=-40, max=40)
    return torch.mean(ssnr)
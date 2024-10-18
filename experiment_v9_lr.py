from model_v9_LR import *
from options import *
from utils import * 
from torchvision import transforms
from mask_dataset import *
import os
import torch.optim as optim
import numpy as np
from lpips import LPIPS
import pickle

def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    transforms_train = transforms.Compose([
        transforms.Resize((128, 128)),
        # transforms.RandomHorizontalFlip(), # data augmentation
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalization
    ])

    transforms_test = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_dir = './CelebA_HQ_facial_identity_dataset'
    mask_dir = './CelebAMask_HQ_facial_identity_dataset'
    train_dataset = CelebAMaskHQ(os.path.join(data_dir, 'train'), os.path.join(mask_dir, 'train'), transforms_train)
    test_dataset = CelebAMaskHQ(os.path.join(data_dir, 'test'), os.path.join(mask_dir, 'test'), transforms_test)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    enc = GigaGAN(3, args.dec_channels, args.latent_size, args.private_classes, lr_alpha=args.lr_alpha).to(device)
    disc = DisNet(args.latent_size, args.hidden_channels, args.private_classes).to(device)
    
    optimizer_enc = optim.AdamW(enc.parameters(), lr=args.enc_lr, weight_decay=1e-2, betas=(0.9, args.enc_beta2), eps=1e-8)
    optimizer_disc = optim.AdamW(disc.parameters(), lr=args.disc_lr, weight_decay=1e-2, betas=(0.9, args.disc_beta2), eps=1e-8)

    # load pre-trained generator
    
    pretrain_save_path = 'logs/Models_mask_v9_adamw_lr0.0001_beta20.990_lr0.0001_beta20.990_penc0.50_aw0.50_alpha0.80_E3600_3600'
    
    pretrained_ckpt = torch.load(pretrain_save_path)
    enc.load_state_dict(pretrained_ckpt['enc_state_dict'])
    optimizer_enc.load_state_dict(pretrained_ckpt['optimizer_enc_state_dict'])
    disc.load_state_dict(pretrained_ckpt['disc_state_dict'])
    optimizer_disc.load_state_dict(pretrained_ckpt['optimizer_disc_state_dict'])

    if args.last_epoch != 0:
        save_path = 'logs/' + 'Models_mask_v9_adamw_lpips_ft_LR_lr%.4f_beta2%.3f_lr%.4f_beta2%.3f_penc%.2f_aw%.2f_alpha%.2f_E%d_%d' % (
                args.enc_lr,
                args.enc_beta2,
                args.disc_lr,
                args.disc_beta2,
                args.prob_train_enc,
                args.adv_weight,
                args.alpha,
                args.last_epoch,
                args.last_epoch)
        last_ckpt = torch.load(save_path)
        enc.load_state_dict(last_ckpt['enc_state_dict'])
        optimizer_enc.load_state_dict(last_ckpt['optimizer_enc_state_dict'])
        disc.load_state_dict(last_ckpt['disc_state_dict'])
        optimizer_disc.load_state_dict(last_ckpt['optimizer_disc_state_dict'])

    mse_loss = nn.MSELoss()
    cross_entropy = nn.CrossEntropyLoss()
    entropy = HLoss()
    torch.hub.set_dir('./cache/torch/hub/checkpoints')
    lpips_loss = LPIPS(net="alex").to(args.device).eval()

    enc.train()
    disc.train()
    epoch_enc_loss = []
    epoch_disc_loss = []
    for epoch in range(args.last_epoch + 1, args.end_epoch + 1):
        for idx, (x, y, x1, y1, z) in enumerate(train_dataloader):
            x = x.to(device)
            y = y.to(device).long()
            x1 = x1.to(device)
            y1 = y1.to(device).long()

            prob = np.random.rand(1)
            if prob > args.prob_train_enc:
                optimizer_enc.zero_grad()
                z0, recon0 = enc(x, y, is_private=False)
                recon1 = enc.decode(z0, y1, is_private=False)

                loss = 0.0
                adv_loss_enc = -entropy(disc(z0))
                loss += args.adv_weight*adv_loss_enc
                recon0_quality = mse_loss(recon0, x)
                recon0_lpips = lpips_loss(recon0, x)
                recon0_lpips = torch.squeeze(recon0_lpips)
                recon0_lpips = torch.mean(recon0_lpips)
                # loss_recon0 = 0.5 * recon0_quality + 0.5 * recon0_lpips
                loss += (1 - args.adv_weight) * 0.25 * recon0_quality
                loss += (1 - args.adv_weight) * 0.25 * recon0_lpips

                recon1_quality = mse_loss(recon1, x1)
                recon1_lpips = lpips_loss(recon1, x1)
                recon1_lpips = torch.squeeze(recon1_lpips)
                recon1_lpips = torch.mean(recon1_lpips)
                # loss_recon1 = 0.5 * recon1_quality + 0.5 * recon1_lpips
                loss += (1 - args.adv_weight) * 0.25 * recon1_quality
                loss += (1 - args.adv_weight) * 0.25 * recon1_lpips
                # loss_enc0 = args.adv_weight*adv_loss_enc + (1-args.adv_weight)*(loss_recon0 +loss_recon1)/2
                loss.backward()
                optimizer_enc.step()

                epoch_enc_loss.append([adv_loss_enc.item(), recon0_quality.item(), recon0_lpips.item(),
                                       recon1_quality.item(), recon1_lpips.item()])
            else:
                optimizer_disc.zero_grad()
                z0  = enc.encode(x)
                d0 = disc(z0)
                loss_disc = cross_entropy(d0, y)
                optimizer_disc.step()
                epoch_disc_loss.append(loss_disc.item())
            
            if idx % args.show_freq == 0 and idx != 0:
                print(f'====> Epoch: {epoch} Batcch: {idx} Average autoencoderr loss:')
                print(epoch_enc_loss[-1])
                print(f'====> Epoch: {epoch} Batcch: {idx} Average Discriminator Non-sensitive loss:')
                print(epoch_disc_loss[-1])
        if epoch % args.save_freq == 0 and epoch != 0:
            ckpt = {
                'enc_state_dict': enc.state_dict(),
                'disc_state_dict': disc.state_dict(),
                'optimizer_enc_state_dict': optimizer_enc.state_dict(),
                'optimizer_disc_state_dict': optimizer_disc.state_dict(),
            }

            save_path = 'logs/' + 'Models_mask_v9_adamw_lpips_ft_LR%d_lr%.4f_beta2%.3f_lr%.4f_beta2%.3f_penc%.2f_aw%.2f_alpha%.2f_E%d_%d' % (
                args.lr_alpha,
                args.enc_lr,
                args.enc_beta2,
                args.disc_lr,
                args.disc_beta2,
                args.prob_train_enc,
                args.adv_weight,
                args.alpha,
                args.end_epoch,
                epoch)
            torch.save(ckpt, save_path)
            log_save_path = 'logs/' + 'Logs_mask_v9_adamw_lpips_ft_LR%d_lr%.4f_beta2%.3f_lr%.4f_beta2%.3f_penc%.2f_aw%.2f_alpha%.2f_E%d_%d' % (
                args.lr_alpha,
                args.enc_lr,
                args.enc_beta2,
                args.disc_lr,
                args.disc_beta2,
                args.prob_train_enc,
                args.adv_weight,
                args.alpha,
                args.end_epoch,
                epoch)
            pickle.dump([epoch_enc_loss, epoch_disc_loss], open(log_save_path, "wb"))

def main(args):
    train(args)


if __name__ == '__main__':
    args = args_parser()
    
    if torch.cuda.is_available():
        args.gpu = True
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    args.last_epoch = 0
    args.end_epoch = 20
    args.save_freq = 10

    args.latent_size = 512
    args.enc_lr = 1e-4
    args.enc_beta2 = 0.99
    args.disc_lr = 1e-4
    args.disc_beta2 = 0.99
    args.batch_size = 16
    print(args)
    for args.lr_alpha in [10, 20, 40, 80]:
    # args.lr_alpha = 40
    
        main(args)
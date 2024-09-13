import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # hyper-parameters arguments
    parser.add_argument('--last_epoch', type=int, default=0, help="start training epochs")
    parser.add_argument('--end_epoch', type=int, default=100, help="end training epochs")

    parser.add_argument('--batch_size', type=int, default=16, help="training batch size")
    
    parser.add_argument('--enc_lr', type=float, default=1e-3, help='learning rate of the autoencoder model')
    parser.add_argument('--enc_beta2', type=float, default=0.999, help="beta2 parameter of autoencoder Adam optimizer")
    parser.add_argument('--disc_lr', type=float, default=1e-3, help='learning rate of the discriminator')
    parser.add_argument('--disc_beta2', type=float, default=0.999, help="beta2 parameter of discriminator Adam optimizer")

    parser.add_argument('--alpha', type=float, default=0.8, help="lambda parameter for reconstruction")
    parser.add_argument('--prob_train_enc', type=float, default=0.5, help="probability of training the VAE")
    parser.add_argument('--adv_weight', type=float, default=0.5, help="weight for the adversarial loss in training the VAE")
    
    # model arguments
    parser.add_argument('--private_classes', type=int, default=307, help="total number of private classes")
    parser.add_argument('--latent_size', type=int, default=2048, help="latent size of the network")
    parser.add_argument('--dec_channels', type=int, default=32, help="channel size")
    parser.add_argument('--hidden_channels', type=int, default=64, help="hidden dimension of thee discriminator")
    parser.add_argument('--classifier_path', type=str, default='', help="path of the pretrained classifier")

    parser.add_argument('--seed', type=int, default=1, help="random seed")

    parser.add_argument('--gpu', type=bool, default=False, help="if GPU")

    parser.add_argument('--save_freq', type=int, default=100, help="frequency of saving log")
    parser.add_argument('--show_freq', type=int, default=50, help="frequency of show image.")

    args = parser.parse_args()
    return args
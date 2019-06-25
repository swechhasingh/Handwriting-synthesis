import torch
import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import plot_stroke
from utils.constants import Global
from utils.dataset import HandwritingDataset
from utils.data_utils import data_denormalization, data_normalization
from models.models import HandWritingPredictionNet, HandWritingSynthesisNet


def argparser():

    parser = argparse.ArgumentParser(description='PyTorch Handwriting Synthesis Model')
    parser.add_argument('--model', type=str, default='synthesis')
    parser.add_argument('--model_path', type=str,
                        default='./saved_models/best_model_synthesis.pt')
    parser.add_argument('--seq_len', type=int, default=400)
    parser.add_argument('--bias', type=float, default=0.0, help='bias')
    parser.add_argument('--char_seq', type=str, default='This is real handwriting')
    parser.add_argument('--text_req', action='store_true')
    parser.add_argument('--prime', action='store_true')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--data_path', type=str, default='./data/')
    args = parser.parse_args()

    return args


def generate_unconditional_seq(model_path, seq_len, device, bias, style, prime):

    model = HandWritingPredictionNet()
    # load the best model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # initial input
    inp = torch.zeros(1, 1, 3)
    inp = inp.to(device)

    batch_size = 1

    initial_hidden = model.init_hidden(batch_size, device)

    print("Generating sequence....")
    gen_seq = model.generate(inp, initial_hidden, seq_len, bias, style, prime)

    return gen_seq


def generate_conditional_sequence(model_path, char_seq, device, char_to_id,
                                  idx_to_char, bias, prime, prime_seq, real_text):
    model = HandWritingSynthesisNet(window_size=len(char_to_id))
    # load the best model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    prime_mask = None
    if prime:
        real_text = np.array(list(real_text + "  "))
        print(''.join(real_text))
        real_text = np.array([[char_to_id[char]
                               for char in real_text]]).astype(np.float32)
        real_text = torch.from_numpy(real_text).to(device)

        prime_mask = torch.ones(real_text.shape).to(device)

    # initial input
    inp = torch.zeros(1, 1, 3)
    inp = inp.to(device)
    char_seq = np.array(list(char_seq + "  "))

    text = np.array([[char_to_id[char] for char in char_seq]]).astype(np.float32)
    text = torch.from_numpy(text).to(device)

    text_mask = torch.ones(text.shape).to(device)

    batch_size = 1

    hidden, window_vector, kappa = model.init_hidden(batch_size, device)

    print("Generating sequence....")
    gen_seq = model.generate(inp, text, text_mask, hidden,
                             window_vector, kappa, bias, is_map=True,
                             prime=prime, prime_style=prime_seq, prime_seq=real_text, prime_mask=prime_mask)

    length = len(text_mask.nonzero())
    print("Input seq: ", ''.join(idx_to_char(
        text[0].detach().cpu().numpy()))[:length])

    phi = torch.cat(model._phi, dim=1).cpu().numpy()
    print(phi.shape)

    return gen_seq, phi[0].T


if __name__ == '__main__':

    args = argparser()

    # fix random seed
    if args.seed:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_path = args.model_path
    model = args.model

    train_dataset = HandwritingDataset(
        args.data_path, split='train', text_req=args.text_req)

    if args.prime:
        strokes = np.load(args.data_path + 'strokes.npy',
                          allow_pickle=True, encoding='bytes')
        with open(args.data_path + 'sentences.txt') as file:
            texts = file.read().splitlines()
        idx = np.random.randint(0, len(strokes))
        # idx = 1446
        print("idx: ", idx)
        real_text = texts[idx]
        style = strokes[idx]  # 4030
        # plot the sequence
        plot_stroke(style, save_name="style_" + str(idx) + ".png")
        print(real_text)
        mean, std, _ = data_normalization(style)
        style = torch.from_numpy(style).unsqueeze(0).to(device)
        print(style.shape)
    else:
        idx = -1
        real_text = None
        style = None

    if model == 'prediction':
        gen_seq = generate_unconditional_seq(
            model_path, args.seq_len, device, args.bias, style=style, prime=args.prime)

    elif model == 'synthesis':
        gen_seq, phi = generate_conditional_sequence(
            model_path, args.char_seq, device, train_dataset.char_to_id,
            train_dataset.idx_to_char, args.bias, args.prime, style, real_text)

        plt.imshow(phi, cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.xlabel("time steps")
        plt.yticks(np.arange(phi.shape[1]), list(
            args.char_seq + "  "), rotation='horizontal')
        plt.margins(0.2)
        plt.subplots_adjust(bottom=0.15)
        plt.savefig("heat_map" + str(idx) + ".png")
        plt.close()
    # denormalize the generated offsets using train set mean and std

    if args.prime:
        gen_seq = data_denormalization(mean, std, gen_seq)
    else:
        gen_seq = data_denormalization(Global.train_mean, Global.train_std, gen_seq)

    # plot the sequence
    plot_stroke(gen_seq[0], save_name="gen_seq_" + str(idx) + ".png")

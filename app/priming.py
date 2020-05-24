import torch
import numpy as np
import sys
import os
# import matplotlib.pyplot as plt

sys.path.append("../")
from utils import plot_stroke
from utils.constants import Global
from utils.dataset import HandwritingDataset
from utils.data_utils import (
    data_denormalization,
    data_normalization,
    valid_offset_normalization,
)
from models.models import HandWritingSynthesisNet
from generate import generate_conditional_sequence


def generate_handwriting(
    char_seq="hello world",
    real_text="",
    style_path="../app/static/mobile/style.npy",
    save_path="",
    app_path="",
    n_samples=1,
    bias=10.0,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = os.path.join(app_path, "../data/")
    model_path = os.path.join(app_path, "../results/synthesis/best_model_synthesis.pt")
    # seed = 194
    # print("seed:",seed)
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # print(np.random.get_state())

    train_dataset = HandwritingDataset(data_path, split="train", text_req=True)

    prime = True
    is_map = False
    style = np.load(style_path, allow_pickle=True, encoding="bytes").astype(np.float32)
    # plot the sequence
    # plot_stroke(style, os.path.join(save_path, "original.png"))

    print("Priming text: ", real_text)
    mean, std, style = data_normalization(style)
    style = torch.from_numpy(style).unsqueeze(0).to(device)
    # style = valid_offset_normalization(Global.train_mean, Global.train_std, style[None,:,:])
    # style = torch.from_numpy(style).to(device)
    print("Priming sequence size: ", style.shape)
    ytext = real_text + " " + char_seq + "  "

    for i in range(n_samples):
        gen_seq, phi = generate_conditional_sequence(
            model_path,
            char_seq,
            device,
            train_dataset.char_to_id,
            train_dataset.idx_to_char,
            bias,
            prime,
            style,
            real_text,
            is_map,
        )
        # if is_map:
        #     plt.imshow(phi, cmap="viridis", aspect="auto")
        #     plt.colorbar()
        #     plt.xlabel("time steps")
        #     plt.yticks(np.arange(phi.shape[0]), list(ytext), rotation="horizontal")
        #     plt.margins(0.2)
        #     plt.subplots_adjust(bottom=0.15)
        #     plt.show()

        # denormalize the generated offsets using train set mean and std
        print("data denormalization...")
        end = style.shape[1]
        # gen_seq[:,:end] = data_denormalization(mean, std, gen_seq[:, :end])
        # gen_seq[:,end:] = data_denormalization(Global.train_mean, Global.train_std, gen_seq[:,end:])
        gen_seq = data_denormalization(Global.train_mean, Global.train_std, gen_seq)
        # plot the sequence
        print(gen_seq.shape)
        # plot_stroke(gen_seq[0, :end])
        plot_stroke(
            gen_seq[0], os.path.join(save_path, "gen_stroke_" + str(i) + ".png")
        )
        print(save_path)

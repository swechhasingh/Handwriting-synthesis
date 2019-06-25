import torch
import numpy as np
from collections import Counter
from torch.utils.data import DataLoader, Dataset
from utils.data_utils import train_offset_normalization, valid_offset_normalization
from utils.constants import Global


class HandwritingDataset(Dataset):
    """Handwriting dataset."""

    def __init__(self, data_path, split='train', text_req=False, debug=False, max_seq_len=300, data_aug=False):
        """
        Args:
            data_path (string): Path to the data folder.
            split (string): train or valid
        """
        self.text_req = text_req
        self.max_seq_len = max_seq_len
        self.data_aug = data_aug

        strokes = np.load(data_path + 'strokes.npy', allow_pickle=True, encoding='bytes')
        with open(data_path + 'sentences.txt') as file:
            texts = file.read().splitlines()

        # list of length of each stroke in strokes
        lengths = [len(stroke) for stroke in strokes]
        max_len = np.max(lengths)
        n_total = len(strokes)

        # Mask
        mask_shape = (n_total, max_len)
        mask = np.zeros(mask_shape, dtype=np.float32)

        # Convert list of str into array of list of chars
        char_seqs = [list(char_seq) for char_seq in texts]
        char_seqs = np.asarray(char_seqs)

        char_lens = [len(char_seq) for char_seq in char_seqs]
        max_char_len = np.max(char_lens)

        # char Mask
        mask_shape = (n_total, max_char_len)  # (6000,64)
        char_mask = np.zeros(mask_shape, dtype=np.float32)

        # Input text array
        inp_text = np.ndarray((n_total, max_char_len), dtype='<U1')
        inp_text[:, :] = ' '

        # Convert list of stroke(array) into ndarray of size(n_total, max_len, 3)
        data_shape = (n_total, max_len, 3)
        data = np.zeros(data_shape, dtype=np.float32)

        for i, (seq_len, text_len) in enumerate(zip(lengths, char_lens)):
            mask[i, :seq_len] = 1.
            data[i, :seq_len] = strokes[i]
            char_mask[i, :text_len] = 1.
            inp_text[i, :text_len] = char_seqs[i]

        # create vocab
        self.id_to_char, self.char_to_id = self.build_vocab(inp_text)
        self.vocab_size = len(self.id_to_char)

        idx_permute = np.random.permutation(n_total)
        data = data[idx_permute]
        mask = mask[idx_permute]
        inp_text = inp_text[idx_permute]
        char_mask = char_mask[idx_permute]

        if debug:
            data = data[:64]
            mask = mask[:64]
            inp_text = inp_text[:64]
            char_mask = char_mask[:64]

        n_train = int(0.9 * data.shape[0])
        self._data = data
        if split == 'train':
            self.dataset = data[:n_train]
            self.mask = mask[:n_train]
            self.texts = inp_text[:n_train]
            self.char_mask = char_mask[:n_train]
            Global.train_mean, Global.train_std, self.dataset = train_offset_normalization(
                self.dataset)

        elif split == 'valid':
            self.dataset = data[n_train:]
            self.mask = mask[n_train:]
            self.texts = inp_text[n_train:]
            self.char_mask = char_mask[n_train:]
            self.dataset = valid_offset_normalization(
                Global.train_mean, Global.train_std, self.dataset)

    def __len__(self):
        return self.dataset.shape[0]

    def idx_to_char(self, id_seq):
        return np.array([self.id_to_char[id] for id in id_seq])

    def char_to_idx(self, char_seq):
        return np.array([self.char_to_id[char] for char in char_seq]).astype(np.float32)

    def build_vocab(self, texts):
        counter = Counter()
        for text in texts:
            counter.update(text)
        unique_char = sorted(counter)
        vocab_size = len(unique_char)

        id_to_char = dict(zip(np.arange(vocab_size), unique_char))
        char_to_id = dict([(v, k) for (k, v) in id_to_char.items()])
        return id_to_char, char_to_id

    def __getitem__(self, idx):

        mask = torch.from_numpy(self.mask[idx])

        if self.text_req:
            input_seq = torch.zeros(self.dataset[idx].shape, dtype=torch.float32)
            input_seq[1:, :] = torch.from_numpy(self.dataset[idx, :-1, :])

            target = torch.from_numpy(self.dataset[idx])
            text = torch.from_numpy(self.char_to_idx(self.texts[idx]))
            char_mask = torch.from_numpy(self.char_mask[idx])
            return (input_seq, target, mask, text, char_mask)
        elif self.data_aug:
            seq_len = len(mask.nonzero())
            start = 0
            end = self.max_seq_len

            if seq_len > self.max_seq_len:
                start = np.random.randint(0, high=seq_len - self.max_seq_len)
                end = start + self.max_seq_len

            stroke = self.dataset[idx, start:end, :]

            input_seq = torch.zeros(stroke.shape, dtype=torch.float32)
            input_seq[1:, :] = torch.from_numpy(stroke[:-1, :])

            target = torch.from_numpy(stroke)
            mask = mask[start:end]

            return (input_seq, target, mask)
        else:
            input_seq = torch.zeros(self.dataset[idx].shape, dtype=torch.float32)
            input_seq[1:, :] = torch.from_numpy(self.dataset[idx, :-1, :])
            target = torch.from_numpy(self.dataset[idx])
            return (input_seq, target, mask)

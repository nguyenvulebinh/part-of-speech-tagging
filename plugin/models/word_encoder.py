import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CharBiGRU(nn.Module):
    def __init__(self, alphabet_size, pretrain_char_embedding, embedding_dim, hidden_dim, dropout, gpu,
                 bidirect_flag=True):
        super(CharBiGRU, self).__init__()
        print("build char sequence feature extractor: GRU ...")
        self.gpu = gpu
        self.hidden_dim = hidden_dim
        if bidirect_flag:
            self.hidden_dim = hidden_dim // 2
        self.char_drop = nn.Dropout(dropout)
        self.char_embeddings = nn.Embedding(alphabet_size, embedding_dim)
        if pretrain_char_embedding is not None:
            self.char_embeddings.weight.data.copy_(torch.from_numpy(pretrain_char_embedding))
        else:
            self.char_embeddings.weight.data.copy_(
                torch.from_numpy(self.random_embedding(alphabet_size, embedding_dim)))
        self.char_lstm = nn.GRU(embedding_dim, self.hidden_dim, num_layers=1, batch_first=True,
                                bidirectional=bidirect_flag)
        if self.gpu:
            self.char_drop = self.char_drop.cuda()
            self.char_embeddings = self.char_embeddings.cuda()
            self.char_lstm = self.char_lstm.cuda()

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def get_last_hiddens(self, input, seq_lengths):
        """
            input:
                input: Variable(batch_size, word_length)
                seq_lengths: numpy array (batch_size,  1)
            output:
                Variable(batch_size, char_hidden_dim)
            Note it only accepts ordered (length) variable, length size is recorded in seq_lengths
        """
        batch_size = input.size(0)
        char_embeds = self.char_drop(self.char_embeddings(input))
        char_hidden = None
        pack_input = pack_padded_sequence(char_embeds, seq_lengths, True)
        char_rnn_out, char_hidden = self.char_lstm(pack_input, char_hidden)
        # char_rnn_out, _ = pad_packed_sequence(char_rnn_out)
        return char_hidden.transpose(1, 0).contiguous().view(batch_size, -1)

    def get_all_hiddens(self, input, seq_lengths):
        """
            input:
                input: Variable(batch_size,  word_length)
                seq_lengths: numpy array (batch_size,  1)
            output:
                Variable(batch_size, word_length, char_hidden_dim)
            Note it only accepts ordered (length) variable, length size is recorded in seq_lengths
        """
        batch_size = input.size(0)
        char_embeds = self.char_drop(self.char_embeddings(input))
        char_hidden = None
        pack_input = pack_padded_sequence(char_embeds, seq_lengths, True)
        char_rnn_out, char_hidden = self.char_lstm(pack_input, char_hidden)
        char_rnn_out, _ = pad_packed_sequence(char_rnn_out)
        return char_rnn_out.transpose(1, 0)

    def forward(self, input, seq_lengths):
        return self.get_last_hiddens(input, seq_lengths)


class CharCNN(nn.Module):
    def __init__(self, alphabet_size, pretrain_char_embedding, embedding_dim, hidden_dim, dropout, gpu=False):
        super(CharCNN, self).__init__()
        print("build char sequence feature extractor: CNN ...")
        self.gpu = gpu
        self.hidden_dim = hidden_dim
        self.char_drop = nn.Dropout(dropout)
        self.char_embeddings = nn.Embedding(alphabet_size, embedding_dim)
        if pretrain_char_embedding is not None:
            self.char_embeddings.weight.data.copy_(torch.from_numpy(pretrain_char_embedding))
        else:
            self.char_embeddings.weight.data.copy_(
                torch.from_numpy(self.random_embedding(alphabet_size, embedding_dim)))
        self.char_cnn = nn.Conv1d(embedding_dim, self.hidden_dim, kernel_size=3, padding=1)
        if self.gpu:
            self.char_drop = self.char_drop.cuda()
            self.char_embeddings = self.char_embeddings.cuda()
            self.char_cnn = self.char_cnn.cuda()

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def get_last_hiddens(self, input, seq_lengths):
        """
            input:
                input: Variable(batch_size, word_length)
                seq_lengths: numpy array (batch_size,  1)
            output:
                Variable(batch_size, char_hidden_dim)
            Note it only accepts ordered (length) variable, length size is recorded in seq_lengths
        """
        batch_size = input.size(0)
        char_embeds = self.char_drop(self.char_embeddings(input))
        char_embeds = char_embeds.transpose(2, 1).contiguous()
        char_cnn_out = self.char_cnn(char_embeds)
        char_cnn_out = F.max_pool1d(char_cnn_out, char_cnn_out.size(2)).view(batch_size, -1)
        return char_cnn_out

    def get_all_hiddens(self, input, seq_lengths):
        """
            input:
                input: Variable(batch_size,  word_length)
                seq_lengths: numpy array (batch_size,  1)
            output:
                Variable(batch_size, word_length, char_hidden_dim)
            Note it only accepts ordered (length) variable, length size is recorded in seq_lengths
        """
        batch_size = input.size(0)
        char_embeds = self.char_drop(self.char_embeddings(input))
        char_embeds = char_embeds.transpose(2, 1).contiguous()
        char_cnn_out = self.char_cnn(char_embeds).transpose(2, 1).contiguous()
        return char_cnn_out

    def forward(self, input, seq_lengths):
        return self.get_last_hiddens(input, seq_lengths)


if __name__ == '__main__':
    char_model = CharCNN(alphabet_size=10,
                         pretrain_char_embedding=None,
                         embedding_dim=32,
                         hidden_dim=32,
                         dropout=0.2,
                         gpu=False)
    embed = char_model(torch.tensor([[0, 1, 3, 2, 1], [0, 1, 3, 2, 6], [0, 1, 1, 1, 1]]), torch.tensor([4, 5, 1]))
    print(embed.shape)

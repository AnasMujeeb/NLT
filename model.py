import torch
import torch.nn as nn

# Batch-friendly Encoder
class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        # batch_first=True taake input shape (batch, seq_len) ho
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, input_batch, hidden_state):
        embedded = self.embedding(input_batch)
        output, hidden_state = self.gru(embedded, hidden_state)
        return output, hidden_state

# Batch-friendly Attention Decoder
class AttnDecoder(nn.Module):
    def __init__(self, hidden_size, vocab_size, max_length=20):
        super(AttnDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.max_length = max_length

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.attn = nn.Linear(hidden_size * 2, max_length)
        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)
        
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input_word, hidden_state, encoder_outputs):
        embedded = self.embedding(input_word) # shape: (batch, 1, hidden_size)
        
        # Attention weights calculation for batch
        attn_weights = torch.softmax(
            self.attn(torch.cat((embedded[:, 0], hidden_state[0]), dim=1)), dim=1
        )
        
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        
        output = torch.cat((embedded[:, 0], attn_applied[:, 0]), dim=1)
        output = self.attn_combine(output).unsqueeze(1)
        output = torch.relu(output)
        
        output, hidden_state = self.gru(output, hidden_state)
        prediction = self.softmax(self.out(output))
        
        return prediction, hidden_state

print("Batch-compatible Attention Model tayar hai!")
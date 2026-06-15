import torch
import torch.nn as nn
from torch import optim
import csv
import os
import random

from vocab_conv import french_vocab, english_vocab
from model import Encoder, AttnDecoder

max_length = 20
hidden_size = 256
batch_size = 64        
learning_rate = 0.001  

def sentence_to_indices(vocab, sentence):
    # 🔥 Training tokenizer clean pipeline
    for char in [".", ",", "!", "?"]:
        sentence = sentence.replace(char, " ")
    tokens = sentence.lower().strip().split()
    
    indexes = []
    for word in tokens:
        indexes.append(vocab.word2index.get(word, 3))
    indexes.append(2) # EOS
    if len(indexes) < max_length:
        indexes += [0] * (max_length - len(indexes))
    return indexes[:max_length]

print("Bada Dataset load ho raha hai...")
pairs = []

if os.path.exists("train.csv"):
    with open("train.csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 2:
                pairs.append((row[1], row[0]))

random.seed(100)
random.shuffle(pairs)
pairs = pairs[:38000]

input_indices = [sentence_to_indices(french_vocab, p[0]) for p in pairs]
target_indices = [sentence_to_indices(english_vocab, p[1]) for p in pairs]

input_tensor = torch.tensor(input_indices, dtype=torch.long)
target_tensor = torch.tensor(target_indices, dtype=torch.long)

encoder = Encoder(french_vocab.n_words, hidden_size)
decoder = AttnDecoder(hidden_size, english_vocab.n_words, max_length)

enc_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
dec_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()

epochs = 15  
n_samples = input_tensor.size(0)

print(f"\n[PURE TRAINING MODE] Running fresh clean training...")
for epoch in range(epochs):
    epoch_loss = 0
    num_batches = 0
    for i in range(0, n_samples, batch_size):
        input_batch = input_tensor[i : i + batch_size]
        target_batch = target_tensor[i : i + batch_size]
        if input_batch.size(0) < batch_size: continue  
            
        enc_optimizer.zero_grad()
        dec_optimizer.zero_grad()
        loss = 0
        hidden = torch.zeros(1, input_batch.size(0), hidden_size)
        encoder_outputs, hidden = encoder(input_batch, hidden)
        dec_input = torch.ones(input_batch.size(0), 1, dtype=torch.long)  
        
        for t in range(max_length):
            output, hidden = decoder(dec_input, hidden, encoder_outputs)
            loss += criterion(output.squeeze(1), target_batch[:, t])
            dec_input = target_batch[:, t].unsqueeze(1)  
            
        loss.backward()
        enc_optimizer.step()
        dec_optimizer.step()
        epoch_loss += loss.item() / max_length
        num_batches += 1
        
    print(f"Epoch {epoch+1}/{epochs} Complete! Loss: {epoch_loss / num_batches:.4f}")
    torch.save(encoder.state_dict(), 'encoder_final.pth')
    torch.save(decoder.state_dict(), 'attn_decoder_final.pth')

print("\n[🏆 PRO SUCCESS] Clean Master Model Trained!")
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
learning_rate = 0.0005  # Controlled LR taake complex combinations balance seekhay

def sentence_to_indices(vocab, sentence):
    for char in [".", ",", "!", "?"]:
        sentence = sentence.replace(char, " ")
    tokens = sentence.lower().strip().split()
    indexes = [vocab.word2index.get(word, 3) for word in tokens]
    indexes.append(2) # EOS
    if len(indexes) < max_length:
        indexes += [0] * (max_length - len(indexes))
    return indexes[:max_length]

print("Base Dataset load ho raha hai...")
base_pairs = []

if os.path.exists("train.csv"):
    with open("train.csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 2:
                fr = row[1].lower().strip()
                en = row[0].lower().strip()
                if 1 <= len(fr.split()) <= 8: # Chote sentences filter kiye matrix capacity k liye
                    base_pairs.append((fr, en))

print(f"[INFO] Base Sentences Loaded: {len(base_pairs)}")

# 🚀 NATIVE DATA AUGMENTATION: Generating Compound Sequences
print("⚡ Generating Native Compound Sentences (Combining with 'et' / 'mais')...")
augmented_pairs = []

# Random selections se high-quality compounds generate karna
random.seed(42)
sample_size = min(len(base_pairs), 15000)
slice_A = random.sample(base_pairs, sample_size)
slice_B = random.sample(base_pairs, sample_size)

for pair1, pair2 in zip(slice_A, slice_B):
    # Clause connection via 'et' (and)
    fr_et = f"{pair1[0]} et {pair2[0]}"
    en_and = f"{pair1[1]} and {pair2[1]}"
    if len(fr_et.split()) < max_length and len(en_and.split()) < max_length:
        augmented_pairs.append((fr_et, en_and))
        
    # Clause connection via 'mais' (but)
    fr_mais = f"{pair1[0]} mais {pair2[0]}"
    en_but = f"{pair1[1]} but {pair2[1]}"
    if len(fr_mais.split()) < max_length and len(en_but.split()) < max_length:
        augmented_pairs.append((fr_mais, en_but))

print(f"[INFO] Generated Synthetic Compound Pairs: {len(augmented_pairs)}")

# All features merge and heavy shuffling
final_training_pool = base_pairs + augmented_pairs
random.shuffle(final_training_pool)
print(f"🔥 Total Augmented Dataset Size For Training: {len(final_training_pool)}")

final_training_pool.sort(key=lambda x: len(x[0].split()))

input_indices = [sentence_to_indices(french_vocab, p[0]) for p in final_training_pool]
target_indices = [sentence_to_indices(english_vocab, p[1]) for p in final_training_pool]

input_tensor = torch.tensor(input_indices, dtype=torch.long)
target_tensor = torch.tensor(target_indices, dtype=torch.long)

encoder = Encoder(french_vocab.n_words, hidden_size)
decoder = AttnDecoder(hidden_size, english_vocab.n_words, max_length)

# Master weights state load ho rahi hy
if os.path.exists('encoder_final.pth') and os.path.exists('attn_decoder_final.pth'):
    print("[INFO] Loading master weights to inject compound structural knowledge...")
    encoder.load_state_dict(torch.load('encoder_final.pth', weights_only=True))
    decoder.load_state_dict(torch.load('attn_decoder_final.pth', weights_only=True))

enc_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
dec_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()

epochs = 12  # 12 rounds system ko completely adapt karwa denge
n_samples = input_tensor.size(0)

print(f"\n[NATIVE COMPOUND MASTER MODE] Training Started...")
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
        
    print(f"Epoch {epoch+1}/{epochs} Complete! Compound-Aware Loss: {epoch_loss / num_batches:.4f}")
    torch.save(encoder.state_dict(), 'encoder_final.pth')
    torch.save(decoder.state_dict(), 'attn_decoder_final.pth')

print("\n[🏆 MISSION ACCOMPLISHED] Model parameters are now natively compound-compatible!")
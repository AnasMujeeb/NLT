import torch
import torch.nn as nn
from torch import optim
import csv
import os

from vocab_conv import french_vocab, english_vocab
from model import Encoder, AttnDecoder

max_length = 20
hidden_size = 256
batch_size = 64        
learning_rate = 0.0003  # Low LR taake pichli seekhi hui perfect grammar kharab na ho

def sentence_to_indices(vocab, sentence):
    indexes = []
    for word in sentence.split():
        indexes.append(vocab.word2index.get(word, 3))
    indexes.append(2)
    if len(indexes) < max_length:
        indexes += [0] * (max_length - len(indexes))
    return indexes[:max_length]

print("Dataset se targeted words (Days, Months, Greetings) nikaale ja rahe hain...")
pairs = []

# Handcrafted core features jo model bhool raha hy
boost_vocabulary = [
    ["bonjour", "hello"],
    ["salut", "hi"],
    ["merci", "thank you"],
    ["bonne chance", "good luck"],
    ["j'ai un ami", "i have a friend"]
]
pairs.extend(boost_vocabulary * 150) # Inko heavy weight dena

# Target keywords jinmein model confuse ho raha hy
target_keywords = [
    'lundi', 'mardi', 'mercredi', 'jeudi', 'vendredi', 'samedi', 'dimanche',
    'janvier', 'février', 'mars', 'avril', 'mai', 'juin', 'juillet', 'août', 
    'septembre', 'octobre', 'novembre', 'décembre', 'bonjour', 'ami'
]

if os.path.exists("train.csv"):
    with open("train.csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for i, row in enumerate(reader):
            if i > 25000: # Sirf pehle 25k sentences ke andar se hi filter karna
                break
            if len(row) >= 2:
                fr = row[1].lower().strip()
                en = row[0].lower().strip()
                
                if 1 <= len(fr.split()) <= 12:
                    # Agar line mein koi bhi day, month ya greeting hy to usay uthao
                    if any(kw in fr for kw in target_keywords):
                        pairs.append((fr, en))
                        # Din aur mahino ko double boost dena taake 'saturday' wala ratta toote
                        if 'chaque' in fr or 'anniversaire' in fr:
                            pairs.extend([(fr, en)] * 5)

print(f"[SUCCESS] Total Targeted Boost Sentences: {len(pairs)}")

# Length sorting for speed
pairs.sort(key=lambda x: len(x[0].split()))

input_indices = [sentence_to_indices(french_vocab, p[0]) for p in pairs]
target_indices = [sentence_to_indices(english_vocab, p[1]) for p in pairs]

input_tensor = torch.tensor(input_indices, dtype=torch.long)
target_tensor = torch.tensor(target_indices, dtype=torch.long)

encoder = Encoder(french_vocab.n_words, hidden_size)
decoder = AttnDecoder(hidden_size, english_vocab.n_words, max_length)

# 🚀 PICHLE 25,000 SENTENCES KA DIMAAG LOAD HO RAHA HAI
if os.path.exists('encoder_conv.pth') and os.path.exists('attn_decoder_conv.pth'):
    print("[INFO] 25,000 sentences wali master memory mil gayi! Fine-tuning shuru...")
    encoder.load_state_dict(torch.load('encoder_conv.pth', weights_only=True))
    decoder.load_state_dict(torch.load('attn_decoder_conv.pth', weights_only=True))
else:
    print("[WARNING] Purane weights nahi mile!")

enc_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
dec_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()

epochs = 6  # Sirf 6 rounds! Data chota hy to mushkil se 1-2 minute lega!
n_samples = input_tensor.size(0)

print(f"\n[TARGETED REFINEMENT MODE] Tuning Confusion Zones...")
for epoch in range(epochs):
    epoch_loss = 0
    num_batches = 0
    
    for i in range(0, n_samples, batch_size):
        input_batch = input_tensor[i : i + batch_size]
        target_batch = target_tensor[i : i + batch_size]
        
        if input_batch.size(0) < batch_size:
            continue  
            
        enc_optimizer.zero_grad()
        dec_optimizer.zero_grad()
        
        loss = 0
        current_batch_size = input_batch.size(0)
        hidden = torch.zeros(1, current_batch_size, hidden_size)
        
        encoder_outputs, hidden = encoder(input_batch, hidden)
        dec_input = torch.ones(current_batch_size, 1, dtype=torch.long)  
        
        for t in range(max_length):
            output, hidden = decoder(dec_input, hidden, encoder_outputs)
            loss += criterion(output.squeeze(1), target_batch[:, t])
            dec_input = target_batch[:, t].unsqueeze(1)  
            
        loss.backward()
        enc_optimizer.step()
        dec_optimizer.step()
        
        epoch_loss += loss.item() / max_length
        num_batches += 1
        
    print(f"Epoch {epoch+1}/{epochs} Complete! Tuning Loss: {epoch_loss / num_batches:.4f}")
    torch.save(encoder.state_dict(), 'encoder_conv.pth')
    torch.save(decoder.state_dict(), 'attn_decoder_conv.pth')

print("\n[🏆 MASTER SUCCESS] Confusion zones fixed successfully!")
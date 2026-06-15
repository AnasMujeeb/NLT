import torch
from old.vocab import french_vocab, english_vocab
from model import Encoder, AttnDecoder

max_length = 20
hidden_size = 256

# Sentence ko fix length aur pad tokens ke sath badalna (Jaise train.py mein kiya)
def sentence_to_indices(vocab, sentence):
    indexes = []
    for word in sentence.split():
        indexes.append(vocab.word2index.get(word, 3)) # 3 = <UNK>
    indexes.append(2) # 2 = <EOS>
    if len(indexes) < max_length:
        indexes += [0] * (max_length - len(indexes)) # 0 = <PAD>
    return indexes[:max_length]

# Model structure khara karein
encoder = Encoder(french_vocab.n_words, hidden_size)
decoder = AttnDecoder(hidden_size, english_vocab.n_words, max_length)

# Naye fast-trained weights load karein
encoder.load_state_dict(torch.load('encoder_weights.pth', weights_only=True))
decoder.load_state_dict(torch.load('attn_decoder_weights.pth', weights_only=True))

encoder.eval()
decoder.eval()

def translate(sentence):
    with torch.no_grad():
        indices = sentence_to_indices(french_vocab, sentence)
        # Inference ke liye Batch Size = 1 ki shape banani hai: (1, max_length)
        input_tensor = torch.tensor([indices], dtype=torch.long)
        
        # Hidden state shape batch ke hisaab se: (1, 1, 256)
        hidden = torch.zeros(1, 1, hidden_size)
        
        # Encoder ab poore sentence ko ek sath parhega (No token loops!)
        encoder_outputs, hidden = encoder(input_tensor, hidden)
        
        # Decoder start token <SOS> = 1, shape: (1, 1)
        dec_input = torch.ones(1, 1, dtype=torch.long)
        decoded_words = []
        
        for t in range(max_length):
            output, hidden = decoder(dec_input, hidden, encoder_outputs)
            
            # Dim=2 par top prediction nikalna kyunke output shape (1, 1, vocab_size) hai
            top_value, top_index = output.data.topk(1, dim=2)
            token_id = top_index.item()
            
            if token_id == 2: # <EOS> aa gaya to ruk jao
                break
            elif token_id != 0: # <PAD> tokens ko translation mein nahi dikhana
                decoded_words.append(english_vocab.index2word[token_id])
                
            dec_input = top_index.squeeze(2) # Shape ko wapas batch input (1, 1) banana
            
        return ' '.join(decoded_words)

print("\n-----------------------------------------")
print("Naya Batch-Compatible Translator Tayar Hai!")
print("Band karne ke liye 'exit' likhein.")
print("-----------------------------------------\n")

while True:
    french_text = input("French likhein: ")
    if french_text.lower() == 'exit':
        break
    
    english_translation = translate(french_text)
    print("English Translation:", english_translation, "\n")
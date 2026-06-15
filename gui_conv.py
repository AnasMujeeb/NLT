import utils.session
import tkinter as tk
from tkinter import messagebox
import torch
import torch.nn as nn
import re
import os  # OS module successfully shamil kar diya

# Nayi isolated modules se imports
from vocab_conv import french_vocab, english_vocab
from model import Encoder, AttnDecoder

max_length = 20
hidden_size = 256

def word_to_idx(vocab, word):
    return vocab.word2index.get(word, 3) # 3 = UNK

def idx_to_word(vocab, idx):
    return vocab.index2word.get(idx, "<unk>")

class TranslationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Machine Translation — Conversational Edition")
        self.root.geometry("750x550")
        self.root.configure(bg="#f4f6f9")

        # UI Layout Elements
        self.title_label = tk.Label(root, text="Conversational FR-to-EN Translator", font=("Segoe UI", 22, "bold"), fg="#1e293b", bg="#f4f6f9")
        self.title_label.pack(pady=20)

        self.input_label = tk.Label(root, text="French Source Text:", font=("Segoe UI", 12, "bold"), fg="#475569", bg="#f4f6f9")
        self.input_label.pack(anchor="w", padx=40) # FIX: x ko padx se replace kiya

        self.input_text = tk.Text(root, height=6, width=70, font=("Segoe UI", 11), bd=1, relief="solid")
        self.input_text.pack(pady=5)

        self.translate_btn = tk.Button(root, text="Translate Sentence", font=("Segoe UI", 12, "bold"), fg="white", bg="#2563eb", activebackground="#1d4ed8", activeforeground="white", relief="flat", width=20, command=self.perform_translation)
        self.translate_btn.pack(pady=15, anchor="e", padx=60) # FIX: x ko padx se replace kiya

        self.output_label = tk.Label(root, text="English Translation:", font=("Segoe UI", 12, "bold"), fg="#475569", bg="#f4f6f9")
        self.output_label.pack(anchor="w", padx=40) # FIX: x ko padx se replace kiya

        self.output_display = tk.Text(root, height=6, width=70, font=("Segoe UI", 11, "bold"), fg="#059669", bg="#f8fafc", bd=1, relief="solid", state="disabled")
        self.output_display.pack(pady=5)

        # Pipeline Loading Layer
        self.load_translation_pipeline()

    def load_translation_pipeline(self):
        try:
            fr_size = french_vocab.n_words
            en_size = english_vocab.n_words

            self.encoder = Encoder(fr_size, hidden_size)
            self.decoder = AttnDecoder(hidden_size, en_size, max_length)

            # Isolated weights loading
            if os.path.exists('encoder_conv.pth') and os.path.exists('attn_decoder_conv.pth'):
                self.encoder.load_state_dict(torch.load('encoder_conv.pth', weights_only=True))
                self.decoder.load_state_dict(torch.load('attn_decoder_conv.pth', weights_only=True))
                self.encoder.eval()
                self.decoder.eval()
            else:
                raise FileNotFoundError("Naye weights ('_conv.pth') nahi mile! Pehle train_conv.py chalayein.")

        except Exception as e:
            messagebox.showerror("Pipeline Load Error", f"An error occurred loading the weights:\n{str(e)}")
            self.root.destroy()

    def perform_translation(self):
        raw_text = self.input_text.get("1.0", tk.END).strip()
        if not raw_text:
            self.update_output_display("")
            return

        clean_text = raw_text.lower().strip()
        clean_text = re.sub(r"([.!?])", r" \1", clean_text)
        tokens = clean_text.split()

        input_ids = [word_to_idx(french_vocab, token) for token in tokens][:max_length - 1]
        padding_needed = max_length - len(input_ids)
        if padding_needed > 0:
            input_ids += [0] * padding_needed

        input_tensor = torch.tensor([input_ids], dtype=torch.long)

        try:
            with torch.no_grad():
                encoder_hidden_init = torch.zeros(1, 1, hidden_size)
                encoder_outputs, encoder_hidden = self.encoder(input_tensor, encoder_hidden_init)

                decoder_input = torch.tensor([[1]], dtype=torch.long) # 1 = SOS
                decoder_hidden = encoder_hidden

                decoded_words = []
                for di in range(max_length):
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                    _, topi = decoder_output.topk(1, dim=2)
                    predicted_idx = topi.item()

                    if predicted_idx == 2: # 2 = EOS
                        break

                    word = idx_to_word(english_vocab, predicted_idx)
                    decoded_words.append(word)
                    decoder_input = topi.squeeze(2)

            translated_sentence = " ".join(decoded_words)
            translated_sentence = re.sub(r"\s+([.!?])", r"\1", translated_sentence)
            translated_sentence = translated_sentence.capitalize()

            self.update_output_display(translated_sentence)

        except Exception as e:
            messagebox.showerror("Inference Error", f"An execution pipeline exception occurred:\n{str(e)}")

    def update_output_display(self, text):
        self.output_display.configure(state="normal")
        self.output_display.delete("1.0", tk.END)
        self.output_display.insert("1.0", text)
        self.output_display.configure(state="disabled")

if __name__ == "__main__":
    root = tk.Tk()
    app = TranslationApp(root)
    root.mainloop()
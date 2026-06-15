import tkinter as tk
from tkinter import messagebox
import torch
import torch.nn as nn
import os
import re

from vocab_conv import french_vocab, english_vocab
from model import Encoder, AttnDecoder

max_length = 20
hidden_size = 256

def word_to_idx(vocab, word):
    return vocab.word2index.get(word, 3)

def idx_to_word(vocab, idx):
    return vocab.index2word.get(idx, "<unk>")

class TranslationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Machine Translation — Pure AI Paragraph Edition")
        self.root.geometry("800x650")
        self.root.configure(bg="#0f172a")

        # UI Styling Header Frame
        top_frame = tk.Frame(root, bg="#1e293b", height=80)
        top_frame.pack(fill="x", side="top")
        top_frame.pack_propagate(False)

        self.title_label = tk.Label(top_frame, text="⚡ NMT Pure Neural Paragraph Engine", font=("Segoe UI", 20, "bold"), fg="#38bdf8", bg="#1e293b")
        self.title_label.pack(pady=20)

        main_container = tk.Frame(root, bg="#0f172a")
        main_container.pack(fill="both", expand=True, padx=40, pady=20)

        self.input_label = tk.Label(main_container, text="French Source (Enter Single Line or Full Paragraph):", font=("Segoe UI", 12, "bold"), fg="#94a3b8", bg="#0f172a")
        self.input_label.pack(anchor="w", pady=(0, 5))

        self.input_text = tk.Text(main_container, height=7, width=70, font=("Segoe UI", 11), bg="#1e293b", fg="#f8fafc", bd=0, insertbackground="white", highlightthickness=1, highlightbackground="#334155")
        self.input_text.pack(fill="x", pady=5)

        btn_frame = tk.Frame(main_container, bg="#0f172a")
        btn_frame.pack(fill="x", pady=15)

        self.translate_btn = tk.Button(btn_frame, text="Neural Inference 🚀", font=("Segoe UI", 12, "bold"), fg="white", bg="#0284c7", activebackground="#0369a1", activeforeground="white", relief="flat", padx=20, pady=8, command=self.perform_translation)
        self.translate_btn.pack(side="right")

        self.output_label = tk.Label(main_container, text="English Neural Output (Sentence-by-Sentence Evaluation):", font=("Segoe UI", 12, "bold"), fg="#94a3b8", bg="#0f172a")
        self.output_label.pack(anchor="w", pady=(10, 5))

        self.output_display = tk.Text(main_container, height=7, width=70, font=("Segoe UI", 11, "bold"), fg="#34d399", bg="#1e293b", bd=0, highlightthickness=1, highlightbackground="#334155", state="disabled")
        self.output_display.pack(fill="x", pady=5)

        self.load_translation_pipeline()

    def load_translation_pipeline(self):
        try:
            self.encoder = Encoder(french_vocab.n_words, hidden_size)
            self.decoder = AttnDecoder(hidden_size, english_vocab.n_words, max_length)

            if os.path.exists('encoder_final.pth') and os.path.exists('attn_decoder_final.pth'):
                self.encoder.load_state_dict(torch.load('encoder_final.pth', weights_only=True))
                self.decoder.load_state_dict(torch.load('attn_decoder_final.pth', weights_only=True))
                self.encoder.eval()
                self.decoder.eval()
        except Exception as e:
            messagebox.showerror("Pipeline Load Error", str(e))

    def eval_pure_model(self, clean_sentence_str):
        # 100% PURE Neural network execution block
        tokens = clean_sentence_str.split()
        if not tokens:
            return ""

        input_ids = [word_to_idx(french_vocab, token) for token in tokens]
        input_ids.append(2) # Native EOS Token
        
        if len(input_ids) < max_length:
            input_ids += [0] * (max_length - len(input_ids))
        input_ids = input_ids[:max_length]

        input_tensor = torch.tensor([input_ids], dtype=torch.long)

        try:
            with torch.no_grad():
                encoder_hidden_init = torch.zeros(1, 1, hidden_size)
                encoder_outputs, encoder_hidden = self.encoder(input_tensor, encoder_hidden_init)
                decoder_input = torch.tensor([[1]], dtype=torch.long)
                decoder_hidden = encoder_hidden

                decoded_words = []
                for di in range(max_length):
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                    _, topi = decoder_output.topk(1, dim=2)
                    predicted_idx = topi.item()
                    if predicted_idx == 2: break
                    decoded_words.append(idx_to_word(english_vocab, predicted_idx))
                    decoder_input = topi.squeeze(2)

            return " ".join(decoded_words).capitalize()
        except Exception:
            return ""

    def perform_translation(self):
        raw_text = self.input_text.get("1.0", tk.END).strip()
        if not raw_text:
            self.update_output_display("")
            return

        # 🚀 STEP-BY-STEP NLP PIPELINE FOR PARAGRAPHS
        # Text ko safe units mein break karna bina model weights ko overload kiye
        normalized_text = raw_text.replace('\n', ' . ')
        raw_chunks = re.split(r'(?<=[.!?])\s+', normalized_text)
        
        final_translated_sentences = []

        for chunk in raw_chunks:
            # Punctuation cleaning before token mapping
            clean_chunk = chunk.lower().strip()
            for char in [".", ",", "!", "?"]:
                clean_chunk = clean_chunk.replace(char, " ")
            
            if clean_chunk.strip():
                # Passing the string to pure neural layers
                translated_clause = self.eval_pure_model(clean_chunk)
                if translated_clause:
                    if not translated_clause.endswith("."):
                        translated_clause += "."
                    final_translated_sentences.append(translated_clause)

        # Output text rendering sheet
        self.update_output_display(" ".join(final_translated_sentences))

    def update_output_display(self, text):
        self.output_display.configure(state="normal")
        self.output_display.delete("1.0", tk.END)
        self.output_display.insert("1.0", text)
        self.output_display.configure(state="disabled")

if __name__ == "__main__":
    root = tk.Tk()
    app = TranslationApp(root)
    root.mainloop()
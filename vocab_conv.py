import csv
import os

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS", 3: "UNK"}
        self.n_words = 4

    def addSentence(self, sentence):
        # 🔥 Punctuation ko remove karna taake clean tokens milein
        for char in [".", ",", "!", "?"]:
            sentence = sentence.replace(char, " ")
        for word in sentence.lower().strip().split():
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

print("[CLEAN] Conversational Dictionary build ho rahi hai...")
french_vocab = Lang("french")
english_vocab = Lang("english")

if os.path.exists("train.csv"):
    with open("train.csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for i, row in enumerate(reader):
            if i > 45000:
                break
            if len(row) >= 2:
                english_vocab.addSentence(row[0])
                french_vocab.addSentence(row[1])

print(f"🔥 French dictionary unique words: {french_vocab.n_words}")
print(f"🔥 English dictionary unique words: {english_vocab.n_words}")
print("[SUCCESS] Clean Vocab Tayar Ho Gayi!")
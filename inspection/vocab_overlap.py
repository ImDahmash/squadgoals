

glove_words = set()
with open('data/dwr/glove.840B.300d.txt', 'r') as glove:
    for line in glove:
        glove_words.add(line.split()[0])

missing = 0
missing_upper = 0
total = 0
with open('data/squad/vocab.dat', 'r') as vocab:
    for word in vocab:
        word = word.strip()
        if word not in glove_words:
            missing += 1
            if word[0] == word.upper()[0]:
                missing_upper += 1
        total += 1

print("Missing GloVe vectors for {} / {} ({:.2f}%) of words in vocab.txt".format(
    missing, total, 100.0*missing / total))
print("of these, {} ({:.2f}%) were uppercase which may explain the previous number"
      .format(missing_upper, 100.0*missing_upper / missing))

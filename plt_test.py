import matplotlib.pyplot as plt
import pickle as pkl


with open('syllable_scores.data', 'rb') as f:
    syllable_scores = pkl.load(f)

with open('bleu_scores.data', 'rb') as f:
    bleu_scores  = pkl.load(f)

plt.scatter(syllable_scores, bleu_scores)
plt.xlabel('Syllable Difference')
plt.ylabel('Bleu Score')
plt.title("Syllable Difference vs Bleu Score")
plt.show()
plt.savefig('figure1.png')
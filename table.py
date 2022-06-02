import matplotlib.pyplot as plt
import pickle as pkl
import save_utils
from os import path

if __name__ == "__main__":
    rhyme_score_start = 4
    syll_score_start = 13
    bleu_score_start = 13
    beams = [4, 8, 16, 32, 64]
    diversity_penalty = 20.0
    rhym_avs = []
    syll_avs = []
    for i in range(5):
        bleu_scores = save_utils.load_version_number("bleu_scores", bleu_score_start+i)
        syll_scores = save_utils.load_version_number("syllable_scores", syll_score_start+i)
        rhyme_scores = save_utils.load_version_number("rhyme_scores", rhyme_score_start+i)
        print("Beam ", str(beams[i]))
        rhym_avs.append(sum(rhyme_scores)/len(rhyme_scores))
        syll_avs.append(sum(syll_scores) / len(rhyme_scores))
        outpath = "graphs/"
        fpath = path.join(outpath, f"rhyme_vs_bleu_beam{beams[i]}.png")
        fig, ax = plt.subplots()
        image = ax.scatter(rhyme_scores, bleu_scores, color='b')
        plt.xlabel("Rhyme Score")
        plt.ylabel("Bleu Score")
        plt.title(f"Rhyme Score vs Bleu Score (num_beams={beams[i]}, diversity penalty={diversity_penalty})")
        plt.draw()
        fig.savefig(fpath)
        plt.close()

        fpath = path.join(outpath, f"syll_vs_bleu_beam{beams[i]}.png")
        fig, ax = plt.subplots()
        image = ax.scatter(syll_scores, bleu_scores, color='b')
        plt.xlabel("Syllable Score")
        plt.ylabel("Bleu Score")
        plt.title(f"Syllable Score vs Bleu Score (num_beams = {beams[i]}), diversity penalty={diversity_penalty})")
        plt.draw()
        fig.savefig(fpath)
        plt.close()
    print("rhyme avg = ", rhym_avs)
    print("syll avg = ", syll_avs)
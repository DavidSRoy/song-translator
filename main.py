# from torch.utils.data import TensorDataset, DataLoader
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, PreTrainedTokenizerFast
import json
import pickle as pkl
from tqdm import tqdm
import nltk
nltk.download('words')
import re
from evaluation import evaluate, getBleuScore, getSyllableScore
import matplotlib.pyplot as plt


def main():
    words = set(nltk.corpus.words.words())  # Words in English Dictionary
    x_test = []
    y_test = []

    with open('spanishval.json') as data_file:
        data = json.load(data_file)
        for i in range(0, len(data)):
            skip = False
            x = data[i]['translation']['en']
            y = data[i]['translation']['es']

            # not sure if necessary for test data
            x = re.sub(r'[,\.;!:?]', '', x)
            y = re.sub(r'[,\.;!:?]', '', y)

            for w in nltk.wordpunct_tokenize(x):
                if w.isalpha() and w.lower() not in words:  # checks if alphanumeric string is in dictionary.
                    skip = True
                    break

            if skip:
                continue

            x_test.append(x)
            y_test.append(y)

    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-one-to-many-mmt", src_lang="en_XX")

    syllable_scores = []
    bleu_scores = []

    for i, sentence_en in tqdm(enumerate(x_test)):
        if i == 30:
            break
        model_inputs = tokenizer(sentence_en, return_tensors="pt")

        # # translate from English to Spanish
        generated_tokens = model.generate(
            **model_inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id["es_XX"],
            num_beams=4,
            num_return_sequences=4
        )

        print(generated_tokens)
        
        best_candidate = None
        best_candidate_score = float('inf')

        sentence_es_actual = y_test[i]
            #sentences

        try:
            for j in range(len(generated_tokens)):
                sentence_es = tokenizer.batch_decode(generated_tokens[j], skip_special_tokens=True)
                
                print(sentence_es)
                print(sentence_es_actual)

                #evaluations.append(evaluate(sentence_en, sentence_es, sentence_es_actual))
                
                print("HERE")
                score = getSyllableScore(sentence_en, sentence_es)
                print("SCORE = ")
                print(score)
                if score < best_candidate_score:
                    best_candidate_score = score
                    best_candidate = sentence_es

                print("HERE1")
            print("HERE2")
            syllable_scores.append(best_candidate_score)


        except: 
            continue
            print("EXCEPT")

        s = ''
        sp = [s.join(best_candidate[i] + ' ') for i in range(len(best_candidate))]
        bleu_score = getBleuScore(sentence_es_actual, ''.join(sp))
        bleu_scores.append(bleu_score)


        print(f'Syllable Score: {best_candidate_score}')
        print(f'Bleu Score: {bleu_score}')
        print(f'Syllable Scores: {syllable_scores}')
        print(f'Bleu Scores: {bleu_scores}')


    with open('syllable_scores.data', 'wb') as f:
        pkl.dump(syllable_scores, f)

    with open('bleu_scores.data', 'wb') as f:
        pkl.dump(bleu_scores, f)

    plt.scatter(syllable_scores, bleu_scores)
    plt.xlabel('Syllable Difference')
    plt.ylabel('Bleu Score')
    plt.title("Syllable Difference vs Bleu Score")
    plt.show()
    plt.savefig('figure1.png')


if __name__ == "__main__":
    main()

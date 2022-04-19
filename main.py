# from torch.utils.data import TensorDataset, DataLoader
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, PreTrainedTokenizerFast
import json
# import pickle
from tqdm import tqdm
import nltk
nltk.download('words')
import re
from evaluation import evaluate
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

    evaluations = []
    for i, sentence_en in tqdm(enumerate(x_test)):
        if i == 30:
            break
        model_inputs = tokenizer(sentence_en, return_tensors="pt")

        # # translate from English to Spanish
        generated_tokens = model.generate(
            **model_inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id["es_XX"]
        )

        sentence_es = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        sentence_es_actual = y_test[i]
        print(sentence_es)
        print(sentence_es_actual)

        try:
            evaluations.append(evaluate(sentence_en, sentence_es, sentence_es_actual))
        except:
            continue

    plt.plot(evaluations)
    plt.ylabel('Evaluation')
    plt.show()


if __name__ == "__main__":
    main()
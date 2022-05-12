# from torch.utils.data import TensorDataset, DataLoader
from save_utils import save_data
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import json
from tqdm import tqdm
import nltk
import re
from evaluation import evaluate, getBleuScore, getSyllableScore
import matplotlib.pyplot as plt

nltk.download('words')

NUM_TO_TRANSLATE = 30
NUM_BEAMS = 4
INPUT_LANG_CODE = "en_XX"
OUTPUT_LANG_CODE = "es_XX"
LOGS_ON = True


def log(inp):
    if LOGS_ON:
        print(inp)


def load_mbart_model_and_tokenizer():
    tokenizer = MBart50TokenizerFast.from_pretrained("TuhinColumbia/spanishpoetrymany",src_lang=INPUT_LANG_CODE)
    model = MBartForConditionalGeneration.from_pretrained("TuhinColumbia/spanishpoetrymany")
    # model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")
    # tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-one-to-many-mmt",
    #                                                  src_lang=INPUT_LANG_CODE)
    return model, tokenizer


def load_model_and_tokenizer(model_type):
    """
    Returns the pretrained model and tokenizer specified
    :param model_type: a string that specifies the model we want to load. Current supported models:
    1) mbart
    :return:
    """
    if model_type == "mbart":
        return load_mbart_model_and_tokenizer()


def generate(input_sentence):
    model_inputs = tokenizer(input_sentence, return_tensors="pt")

    # # translate from English to Spanish
    output_ids = model.generate(
        **model_inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[OUTPUT_LANG_CODE]
        # num_beams=NUM_BEAMS,
        # num_return_sequences=NUM_BEAMS
    )
    
    best_candidate = []
    best_candidate_score = float('inf')
    try:
        for j in range(len(output_ids)):
            output_sentence = tokenizer.batch_decode(output_ids[j], skip_special_tokens=True)
            score = getSyllableScore(input_sentence, output_sentence)

            processed = process_candidate(output_sentence)
            print(f"Candidate: {processed}")

            if score < best_candidate_score:
                best_candidate_score = score
                best_candidate = output_sentence
    except (Exception,):
        print("EXCEPT")

    return best_candidate, best_candidate_score


def load_json_test_data(data_path, words):
    x_test = []
    y_test = []
    with open(data_path) as data_file:
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
    log("x_test length "+str(len(x_test)))
    log("y_test length " + str(len(y_test)))
    return x_test, y_test

def process_candidate(c):
    s = ''
    sp = [s.join(c[i] + ' ') for i in range(len(c))]
    processed = ''.join(sp)
    return processed

def translate_and_evaluate(x, y):
    bleu_scores = []
    syllable_scores = []
    for i, original_sentence in tqdm(enumerate(x)):
        if i == NUM_TO_TRANSLATE:
            break
        human_translation = y[i]
        best_candidate, best_candidate_score = generate(original_sentence)
        s = ''
        sp = [s.join(best_candidate[i] + ' ') for i in range(len(best_candidate))]
        bleu_score = getBleuScore(human_translation, ''.join(sp))
        syllable_scores.append(best_candidate_score)
        bleu_scores.append(bleu_score)

        print(f'Syllable Score: {best_candidate_score}')
        print(f'Bleu Score: {bleu_score}')
        print(f'Syllable Scores: {syllable_scores}')
        print(f'Bleu Scores: {bleu_scores}')

    save_data("syllable_scores", syllable_scores)
    save_data("bleu_scores", bleu_scores)

    plt.scatter(syllable_scores, bleu_scores)
    plt.xlabel('Syllable Difference')
    plt.ylabel('Bleu Score')
    plt.title("Syllable Difference vs Bleu Score")
    plt.show()
    plt.savefig('figure1.png')


def translate_EMNLP_data(words):
    x_test, y_test = load_json_test_data("spanishval.json", words)
    translate_and_evaluate(
        x=x_test,
        y=y_test
    )


def translate_parallel_text_data(original_text_path):
    en_lines = []
    es_lines = []

    with open(original_text_path, 'rb') as f:
        en_lines = f.readlines()

    print(en_lines)
    for en in en_lines:
        en_s = str(en)
        en_s = en_s[2:-3]
        print(en_s)
        best = generate(str(en_s))
        es = process_candidate(best)
        es_lines.append(es)

    for es in es_lines:
        print(es)


# load model and tokenizer in outermost scope
model, tokenizer = load_model_and_tokenizer("mbart")


def main():
    words = set(nltk.corpus.words.words())  # Words in English Dictionary
    translate_EMNLP_data(words)
    # translate_parallel_text_data('song_en.txt')


if __name__ == "__main__":
    main()

    # while True:
    #     en = input("EN: ")
    #     best = generate(en)[0]

    #     es = process_candidate(best)
    #     print(f'ES: {es}')
    #     print()



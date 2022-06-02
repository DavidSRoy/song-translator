# from torch.utils.data import TensorDataset, DataLoader
from save_utils import save_data
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import json
from tqdm import tqdm
import nltk
import re
from evaluation import getBleuScore, getSyllableScore, getRhymeScore
import matplotlib.pyplot as plt

nltk.download('words')

NUM_TO_TRANSLATE = 30
NUM_BEAMS = 4
INPUT_LANG_CODE = "es_XX"
OUTPUT_LANG_CODE = "en_XX"
LOGS_ON = True


def log(inp):
    if LOGS_ON:
        print(inp)


def load_mbart_model_and_tokenizer():
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt",
                                                     src_lang=INPUT_LANG_CODE)
    return model, tokenizer


def load_fine_tuned_model():
    tokenizer = MBart50TokenizerFast.from_pretrained("TuhinColumbia/spanishpoetrymany")
    model = MBartForConditionalGeneration.from_pretrained("TuhinColumbia/spanishpoetrymany")
    tokenizer.src_lang = "es_XX"
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
    elif model_type == "fine_tuned":
        return load_fine_tuned_model()


def generate(input_sentence):
    model_inputs = tokenizer(input_sentence, return_tensors="pt")

    # # translate from English to Spanish
    output_ids = model.generate(
        **model_inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[OUTPUT_LANG_CODE],
        num_beams=NUM_BEAMS,
        num_beam_groups=NUM_BEAMS // 2,
        num_return_sequences=NUM_BEAMS,
        diversity_penalty=2.0
    )
    
    best_candidate = []
    best_candidate_score = float('inf')
    try:
        log("------------------------------")
        log("Iinput Sentence: "+input_sentence)
        output_sentences = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        for j in tqdm(range(len(output_ids))):
            output_joined = output_sentences[j]
            output_joined = output_joined.strip()
            score = getSyllableScore(input_sentence, strip_punct(output_joined))
            if score < best_candidate_score:
                best_candidate_score = score
                best_candidate = output_joined
    except (Exception,):
        print("EXCEPT")
    log("Best Candidate: " + best_candidate)
    log("Best Candidate Syll Score: " + str(best_candidate_score))
    log("------------------------------")
    return best_candidate, best_candidate_score


def strip_punct(sent):
    return re.sub(r'[,\.;!:?]', '', sent)

def add_new_line(sentence):
    return re.sub(r'[\.;!:?]', '\n', sentence)


def load_json_test_data(data_path):
    x_test = []
    y_test = []
    with open(data_path) as data_file:
        data = json.load(data_file)
        for i in range(0, len(data)):
            skip = False
            x = data[i]['translation']['es']
            y = data[i]['translation']['en']

            # not sure if necessary for test data
            x = re.sub(r'[,\.;!:?]', '', x)
            y = re.sub(r'[,\.;!:?]', '', y)

            x_test.append(x)
            y_test.append(y)
    log("x_test length "+str(len(x_test)))
    log("y_test length " + str(len(y_test)))
    return x_test, y_test


def load_test_data(data_path_x, data_path_y):
    '''
    :param data_path_x: 23 Spanish Poems
    :param data_path_y: 23 Gold Standard English Translations
    :return: Lists of Spanish and English Poems with punctuation removed.
    '''
    x_test = []
    y_test = []
    with open(data_path_x) as data_file_x, open(data_path_y) as data_file_y:
        data_x = data_file_x.read()
        # data_x.strip('\n')
        data_x = data_x.split("--------------------------------------------------------------------------------")
        data_y = data_file_y.read()
        # data_y.strip('\n')
        data_y = data_y.split("--------------------------------------------------------------------------------")

        for i in range(0, len(data_x)):
            xL = []
            for ln in data_x[i].splitlines():
                xL.append(ln)
                xL.append('\n')

            yL = []
            for ln in data_y[i].splitlines():
                yL.append(ln)
                yL.append('\n')

            x = ' '.join(xL)
            y = ' '.join(yL)
            # x = ' '.join(data_x[i].splitlines())
            # y = ' '.join(data_y[i].splitlines())

            # x = re.sub(r'[,\.;!:?¿]', '', x)
            # y = re.sub(r'[,\.;!:?¿]', '', y)

            x_test.append(x)
            y_test.append(y)
            # log("x_test length "+str(len(x_test)))
            # log("y_test length " + str(len(y_test)))
        return x_test, y_test


def translate_and_evaluate(x, y):
    bleu_scores = []
    syllable_scores = []
    rhyme_scores = []
    for i, original_sentence in tqdm(enumerate(x)):
        if i == NUM_TO_TRANSLATE:
            break
        human_translation = y[i]
        best_candidate, best_candidate_score = generate(original_sentence)
        bleu_score = getBleuScore(human_translation, best_candidate)
        rhyme_score = getRhymeScore(original_sentence, add_new_line(best_candidate))
        syllable_scores.append(best_candidate_score)
        bleu_scores.append(bleu_score)
        rhyme_scores.append(rhyme_score)

        log(f'Bleu Score: {bleu_score}')
        log(f'Rhyme Score: {rhyme_score}')
        log('')
        log(f'Syllable Scores: {syllable_scores}')
        log(f'Bleu Scores: {bleu_scores}')
        log(f'Rhyme Scores: {rhyme_scores}')

    save_data("syllable_scores", syllable_scores)
    save_data("bleu_scores", bleu_scores)
    save_data("rhyme_scores", rhyme_scores)

    plt.scatter(syllable_scores, bleu_scores)
    plt.xlabel('Syllable Difference')
    plt.ylabel('Bleu Score')
    plt.title("Syllable Difference vs Bleu Score")
    plt.show()
    plt.savefig('figure1.png')

    plt.scatter(syllable_scores, rhyme_scores)
    plt.xlabel('Syllable Difference')
    plt.ylabel('Rhyme Score')
    plt.title("Rhyme Score vs Syllable Difference")
    plt.show()
    plt.savefig('rhyme_vs_syllable.png')


def translate_EMNLP_data():
    x_test, y_test = load_json_test_data("data/spanishval.json")
    translate_and_evaluate(
        x=x_test,
        y=y_test
    )


def translate_song(file):
    with open(file, 'r') as f: 
        output = generate(f.read())
    
    with open('output.txt','w') as o:
        o.write(add_new_line(output[0]))
        o.close()
    return output


def translate_poem_data():
    x_test, y_test = load_test_data("data/testspanish.txt", "data/testspanishgold.txt")
    translate_and_evaluate(
        x=x_test,
        y=y_test
    )


# load model and tokenizer in outermost scope
model, tokenizer = load_model_and_tokenizer("fine_tuned")


def main():
    #translate_poem_data()
    translated_song = translate_song('song_en.txt')
    print(translated_song)


if __name__ == "__main__":
    main()

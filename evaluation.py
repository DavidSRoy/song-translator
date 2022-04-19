import pronouncing as pro
from syltippy import syllabize
from nltk.translate.bleu_score import sentence_bleu

SYLLABLE_WEIGHT = 0.6
BLEU_WEIGHT = 0.4
SYLLABLE_THRESHOLD = 10  # to be removed

def getNumSyllablesEN(sentence):
    lst = sentence.split(' ')
    total_ct = 0
    for word in lst:

        # try:
        pro_list = pro.phones_for_word(word)
        total_ct += pro.syllable_count(pro_list[0])
        # except:
        #     print(word)

    return total_ct


def getNumSyllablesES(word):
    return len(syllabize(word)[0])


def getBleuScore(reference, candidate):
    return sentence_bleu(reference, candidate)


def evaluate(sentence_en, sentence_es, sentence_es_actual):
    syllable_score = getNumSyllablesEN(sentence_en) - getNumSyllablesES(sentence_es)
    bleu_score = getBleuScore(sentence_es, sentence_es_actual)

    print(f' {sentence_es} + {sentence_es_actual}')
    print(f'Syllable Score: {syllable_score}')
    print(f'Bleu Score: {bleu_score}')

    return (-pow(syllable_score, 2) * SYLLABLE_WEIGHT + SYLLABLE_THRESHOLD) + bleu_score * BLEU_WEIGHT


# while True:
#     sentence_en = input("Enter an English sentence: ")
#     sentence_es = input("Enter a candidate Spanish sentence: ")
#     sentence_es_actual = input("Enter the actual Spanish translation")
#     score = evaluate(sentence_en, sentence_es, sentence_es_actual)
#     print(f'Score: {score}')


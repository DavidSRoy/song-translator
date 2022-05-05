import pronouncing as pro
from syltippy import syllabize
from nltk.translate.bleu_score import sentence_bleu

SYLLABLE_WEIGHT = 0.6
BLEU_WEIGHT = 0.4
SYLLABLE_THRESHOLD = 10  # to be removed


def getNumSyllablesEN(sentence):
    # print(type(sentence))
    # print(sentence)
    lst = sentence.split(' ')
    print(lst)
    total_ct = 0
    for word in lst:
        pro_list = pro.phones_for_word(word)
        total_ct += pro.syllable_count(pro_list[0])
        print("succ")

    return total_ct


def getNumSyllablesES(word):
    return len(syllabize(word)[0])


def getNumSyllablesESSentence(sentence):
    score = 0
    for word in sentence:
        score += len(syllabize(word)[0])
    return score


def getBleuScore(reference, candidate):
    print(reference)
    return sentence_bleu(reference, candidate)


def getSyllableScore(sentence_es, sentence_en):
    syllable_score = abs(getNumSyllablesEN(sentence_en) - getNumSyllablesESSentence(sentence_es))
    return syllable_score


def evaluate(sentence_es, sentence_en, sentence_en_actual):
    syllable_score = getSyllableScore(sentence_en, sentence_es)
    bleu_score = getBleuScore(sentence_es, sentence_en_actual)

    print(f' {sentence_es} + {sentence_es_actual}')
    print(f'Syllable Score: {syllable_score}')
    print(f'Bleu Score: {bleu_score}')

    return (-pow(syllable_score, 2) * SYLLABLE_WEIGHT + SYLLABLE_THRESHOLD) + bleu_score * BLEU_WEIGHT


if __name__ == "__main__":

    while True:
        sentence_en = input("Enter an English sentence: ")
        sentence_es = input("Enter a candidate Spanish sentence: ")
        sentence_es_actual = input("Enter the actual Spanish translation")
        print(f"SYLLABLES ES = {getNumSyllablesES(sentence_es)}")
        print(f"SYLLABLES ES ACTUAL = {getNumSyllablesES(sentence_es_actual)}")
        score = evaluate(sentence_en, sentence_es, sentence_es_actual)
        print(f'Score: {score}')


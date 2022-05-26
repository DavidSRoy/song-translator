import pronouncing as pro
from syltippy import syllabize
from nltk.translate.bleu_score import sentence_bleu

SYLLABLE_WEIGHT = 0.6
BLEU_WEIGHT = 0.4
SYLLABLE_THRESHOLD = 10  # to be removed


def getNumSyllablesEN(sentence):
    lst = sentence.split(' ')
    total_ct = 0
    truncated_word = ""
    last_word = ""
    for word in lst:
        try:
            if truncated_word == "":
                pro_list = pro.phones_for_word(word)
                total_ct += pro.syllable_count(pro_list[0])
                last_word = word
            else:
                attempt_suffix = truncated_word + word
                try:
                    pro_list = pro.phones_for_word(attempt_suffix)
                    syll_count = pro.syllable_count(pro_list[0])
                    total_ct += syll_count
                    last_word = truncated_word
                    truncated_word = ""
                except:
                    attempt_suffix = last_word + attempt_suffix
                    try:
                        pro_list = pro.phones_for_word(attempt_suffix)
                        syll_count = pro.syllable_count(pro_list[0])
                        total_ct -= pro.syllable_count(pro.pronunciations(last_word))
                        total_ct += syll_count
                        print(attempt_suffix)
                        last_word = attempt_suffix
                        truncated_word = ""
                    except:
                        truncated_word = truncated_word + word
        except:
            truncated_word = word

    return total_ct


def getNumSyllablesES(word):
    return len(syllabize(word)[0])


def getNumSyllablesESSentence(sentence):
    score = 0
    sentence = sentence.split(" ")
    for word in sentence:
        score += len(syllabize(word)[0])
    return score


def getBleuScore(reference, candidate):
    print(reference)
    return sentence_bleu([reference], candidate)


def getSyllableScore(sentence_es, sentence_en):
    english_syll_count = getNumSyllablesEN(sentence_en)
    spanish_syll_count = getNumSyllablesESSentence(sentence_es)
    print("English syllable score: "+str(english_syll_count))
    print("Spanish syllable score: "+str(spanish_syll_count))
    syllable_score = abs(english_syll_count - spanish_syll_count)
    return syllable_score


def evaluate(sentence_en, sentence_es, sentence_es_actual):
    syllable_score = getSyllableScore(sentence_en, sentence_es)
    bleu_score = getBleuScore(sentence_es, sentence_es_actual)

    print(f' {sentence_es} + {sentence_es_actual}')
    print(f'Syllable Score: {syllable_score}')
    print(f'Bleu Score: {bleu_score}')

    return (-pow(syllable_score, 2) * SYLLABLE_WEIGHT + SYLLABLE_THRESHOLD) + bleu_score * BLEU_WEIGHT


if __name__ == "__main__":
    while True:
        sentence_en = input("Enter an Spanish sentence: ")
        print(getNumSyllablesESSentence(sentence_en))


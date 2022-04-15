import pronouncing as pro
from syltippy import syllabize


print('hello')




def getNumSyllablesEN(sentence):
    lst = sentence.split(' ')
    total_ct = 0
    for word in lst:
        pro_list = pro.phones_for_word(word)
        total_ct += pro.syllable_count(pro_list[0])
    return total_ct

def getNumSyllablesES(word):
    return len(syllabize(word)[0])


def evaluate(sentence_en, sentence_es):
    return getNumSyllablesEN(sentence_en) - getNumSyllablesES(sentence_es)


while True:
    sentence_en = input("Enter an English sentence: ")
    sentence_es = input("Enter an Spanish sentence: ")
    score = evaluate(sentence_en, sentence_es)
    print(f'Score: {score}')


# while True:
#     word = input("Enter a word: ")
#     num = getNumSyllablesES(word)
#     print(f'Number of syllables: {num}')
    
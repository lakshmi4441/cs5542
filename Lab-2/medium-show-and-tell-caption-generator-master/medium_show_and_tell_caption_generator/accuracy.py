import nltk
from itertools import zip_longest
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu


def accuracy():
    imageCount = 0
    with open("../pred.txt") as f2,  open("../true.txt") as f1:
        for line, line1 in zip_longest(f1, f2):
            y_true = word_tokenize(line)
            y_pred = word_tokenize(line1)
            BLEUscore = sentence_bleu(y_true, y_pred, weights=(1,0,0,0))
            print("BLEU Accuracy for Image",imageCount, "is",BLEUscore)
            imageCount = imageCount + 1


if __name__ == "__main__":
    accuracy()

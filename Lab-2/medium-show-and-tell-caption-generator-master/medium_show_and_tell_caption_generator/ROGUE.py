from PyRouge.pyrouge import Rouge
from itertools import zip_longest

r = Rouge()

imageCount = 0
with open("../pred.txt") as f2,  open("../true.txt") as f1:
    for line, line1 in zip_longest(f1, f2):
        system_generated_summary = line
        manual_summmary = line1
        [precision, recall, f_score] = r.rouge_l([system_generated_summary], [manual_summmary])
        print("Precision is of Image "+ str(imageCount)+":" + str(precision) + "\nRecall of Image "+ str(imageCount)+":" + str(recall) + "\nF Score Image "+ str(imageCount)+":" + str(f_score))
        imageCount = imageCount + 1


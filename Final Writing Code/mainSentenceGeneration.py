from multilineSentences import generateSentence
from ThinningAlgorithm import makingImageThinned

# *****************************************************************
sentence = "Something is afoot and its not at the end of my leg"
hand_writer = "dewaldCapital"
# *****************************************************************

generateSentence(sentence, hand_writer)
makingImageThinned(sentence + " _" + hand_writer + ".jpg")




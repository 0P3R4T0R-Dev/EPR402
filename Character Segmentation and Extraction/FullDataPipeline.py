from HelperFunctions import *
from viewH5 import viewH5

num_samples = 4000
num_pixels = 35 ** 2

# filenames = ["Keegan_BigParagraph", "Keegan_Sentences", "Stefan_BigParagraph", "Stefan_Sentences", "Johan_BigParagraph",
#              "Johan_Sentences", "David_BigParagraph", "David_Sentences", "StefanP_BigParagraph", "StefanP_Sentences",
#              "Brian_BigParagraph", "Brian_Sentences"]
# filenames = ['Myburgh_BigParagraph', 'Myburgh_Sentences']
filenames = ['Rynhard_BigParagraph', 'Rynhard_Sentences', 'Tristan_BigParagraph', 'Tristan_Sentences',
             'Dewald_BigParagraph', 'Dewald_Sentences', 'TristanP_BigParagraph', 'TristanP_Sentences',
             'Viren_BigParagraph', 'Viren_Sentences']

for filename in filenames:
    generateGrids(filename, num_samples, debug=False)


person = ["Stefan_BigParagraph", "Johan_BigParagraph", "David_BigParagraph", "StefanP_BigParagraph",
          "Brian_BigParagraph", "Keegan_BigParagraph", "Tristan_BigParagraph", "Rynhard_BigParagraph",
          "TristanP_BigParagraph", "Viren_BigParagraph", "Dewald_BigParagraph"]
ID = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
makeCSV(person, ID, "BigParagraph_11People", num_pixels)

person = ["Stefan_Sentences", "Johan_Sentences", "David_Sentences", "StefanP_Sentences", "Brian_Sentences",
          "Keegan_Sentences", "Tristan_Sentences", "Rynhard_Sentences", "TristanP_Sentences", "Viren_Sentences",
          "Dewald_Sentences"]
ID = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
makeCSV(person, ID, "Sentences_11People", num_pixels)

person = ["Rynhard_Sentences"]
ID = ["7"]
makeCSV(person, ID, "Rynhard_Sentences", num_pixels)

person = ["Tristan_Sentences"]
ID = ["6"]
makeCSV(person, ID, "Tristan_Sentences", num_pixels)

person = ["Dewald_Sentences"]
ID = ["10"]
makeCSV(person, ID, "Dewald_Sentences", num_pixels)

person = ["TristanP_Sentences"]
ID = ["8"]
makeCSV(person, ID, "TristanP_Sentences", num_pixels)

person = ["Viren_Sentences"]
ID = ["9"]
makeCSV(person, ID, "Viren_Sentences", num_pixels)

# person = ['Rynhard_Sentences']
# ID = ["5"]
# makeCSV(person, ID, "Myburgh_Sentences", num_pixels)
#
# person = ["Stefan_BigParagraph", "Johan_BigParagraph", "David_BigParagraph",
#           "StefanP_BigParagraph", "Brian_BigParagraph", "Myburgh_BigParagraph"]
# ID = ["0", "1", "2", "3", "4", "5"]
# makeCSV(person, ID, "BigParagraph_6thPerson_Myburgh", num_pixels)
# person = ["Stefan_BigParagraph", "Johan_BigParagraph", "David_BigParagraph",
#           "StefanP_BigParagraph", "Brian_BigParagraph"]
# ID = ["0", "1", "2", "3", "4"]
# makeCSV(person, ID, "BigParagraph_5thPerson", num_pixels)
#
#
# person = ["Stefan_Sentences", "Johan_Sentences", "David_Sentences", "StefanP_Sentences", "Brian_Sentences"]
# ID = ["0", "1", "2", "3", "4"]
# makeCSV(person, ID, "Sentences_5thPerson", num_pixels)

# viewH5("G:/My Drive/TrainingDataEPR/BigParagraph_5thPerson1225.h5")



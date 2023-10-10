from CameraControl import takePicture
from CharacterSegmentation import extract_and_label_letters

# *****************************************************************
filename = "dewaldFox"
nameOfUser = "temp"
sentence = "thequickbrownfoxjumpsoverthelazydog"
# *****************************************************************

takePicture(filename + ".jpg")
extract_and_label_letters(filename, "../../FONTS/" + nameOfUser, sentence)





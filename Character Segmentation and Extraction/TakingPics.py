import cv2
from PIL import Image

filename = "Dewald_Sentences.jpg"

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
width = 1920
height = 1080
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
flag = True
if not cam.isOpened():
    print("Cannot open camera")
    exit()
while flag:
    # Capture frame-by-frame
    ret, frame = cam.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    grayVersionOfFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Display the resulting frame
    # cv2.imshow('frame', frame)
    frameSmaller = cv2.resize(frame, (960, 540))
    cv2.imshow(filename, frameSmaller)
    if cv2.waitKey(1) == ord('q'):
        flag = False
        cv2.imwrite(filename, frame)
        break

cam.release()
cv2.destroyAllWindows()

image = Image.open(filename)
image.show()

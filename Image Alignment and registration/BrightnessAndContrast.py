import numpy as np
import cv2


def cvVersion(alpha=0, beta=255):
    # contrast is the distance between alpha and beta
    # brightness is the up/down shift of alpha and beta
    alpha = -55  # start at 0
    beta = 200  # start at 255
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        cv2.normalize(frame, frame, alpha, beta, cv2.NORM_MINMAX)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def numpyVersion():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        contrast = 1.25
        brightness = 50
        frame[:, :, 2] = np.clip(contrast * frame[:, :, 2] + brightness, 0, 255)
        frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


cvVersion()

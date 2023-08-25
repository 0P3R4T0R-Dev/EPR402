import cv2
import time

start_time = time.time()
img = cv2.imread("test.jpg")
sr = cv2.dnn_superres.DnnSuperResImpl_create()
path = "EDSR_x4.pb"
sr.readModel(path)
sr.setModel("edsr", 4)  # set the model by passing the value and the upsampling ratio
result = sr.upsample(img)  # upscale the input image
cv2.imwrite("test upscaled with EDSR.jpg", result)
end_time = time.time()
print("Time taken to upscale image with EDSR: ", end_time - start_time)

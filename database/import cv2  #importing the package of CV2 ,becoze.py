import cv2  #importing the package of CV2 ,becoze image processing is done computer vision module
import numpy as np #importing the package of numpy,becoze commonly used packages for scientific computing in Python

# Load Yolo
net = cv2.dnn.readNet("weights/yolov3.weights", "cfg/yolov3.cfg") # used to load the weight and cfg file.
classes = [] #declaration of array class
with open("coco.names", "r") as f: 
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames() #net.getLayerNames(): It gives you list of all layers used in a network. Like I am currently working with yolov3. It gives me a list of 254 layers.
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
img = cv2.imread("src_room.jpg") #image is uploaded here,for the processing
img = cv2.resize(img, None, fx=0.8, fy=0.8)#image resizing is done like height and weight
height, width, channels = img.shape

# Detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

net.setInput(blob)
outs = net.forward(output_layers)

# Showing informations on the screen
class_ids = []#declaration of array class
confidences = []#declaration of array class
boxes = []#declaration of array class
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]#The confidence score indicates how sure the model is that the box contains an object and also how accurate it thinks the box is that predicts. 
        if confidence > 0.5:
            # Object detected,object will be detected if the condition is satisfied.
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates,anchor is appeared with rectanugaler box
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4) #To avoid many boxes we are picking up the boxes with high confidence
print(indexes)
font = cv2.FONT_HERSHEY_PLAIN #font which is help to display the name after detection
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)# her reactangular box appears of the objected detected
        cv2.putText(img, label, (x, y + 30), font, 3, color, 3)#the name of the objected is labbeled


cv2.imshow("Image", img)#it show the image output
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import numpy as np

# Loading YoloV3

cap = cv2.VideoCapture(0)

net = cv2.dnn.readNet('model/yolov3.weights', 'model/yolo3.cfg')
classes = []

with open('model/coco.names', 'r') as f:
	# reading all names to 'classes' variable!
	classes = [line.strip() for line in f.readlines()]
	print('Number of classes:', len(classes))
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]

while True:
	_, frame = cap.read()

	frame = cv2.flip(frame, 1)

	# frame = cv2.imread('sample.jpg')
	frame = cv2.resize(frame, (416,416))

	width, height, channel = frame.shape

	# detecting a object
	blob = cv2.dnn.blobFromImage(frame, 0.00292, (416, 416), (0,0,0), True, crop=False)
	net.setInput(blob)
	outs = net.forward(output_layers)

	class_ids = []
	confidences = []
	boxes = []

	for out in outs:
		for detection in out:
			scores = detection[5:]
			class_id = np.argmax(scores)
			confidence = scores[class_id]
			if confidence > 0.5 :
				# object location
				center_x =  int(detection[0] * width)
				center_y = int(detection[1] * height)
				w = int(detection[2] * width)
				h = int(detection[3] * height)

				# Rectangle coordinates
				x = int(center_x - w/2)
				y = int(center_y - h/2)

				# factor for scale to original size of frame


				boxes.append([x, y, w, h])
				confidences.append(float(confidence))
				class_ids.append(class_id)

	indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
	colors = np.random.uniform(0, 255, size=(len(classes), 3))

	font = cv2.FONT_HERSHEY_PLAIN
	for i in range(len(boxes)):
	    if i in indexes:
	        x, y, w, h = boxes[i]
	        label = str(classes[class_ids[i]])
	        rec_color = colors[i]
	        text_color = colors[i+1]
	        cv2.rectangle(frame, (x, y), (x + w, y + h), rec_color, 1)
	        cv2.putText(frame, label, (x, y + 10), 1, 1, (0,0,255), 1)
	cv2.imshow('frame', frame)

	if cv2.waitKey(1) == 27:
		break
cap.release()
cv2.destroyAllWindows()
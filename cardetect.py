import cv2
import numpy as np
import os

main_path = os.getcwd()

weight_path = os.path.join(main_path, 'yolov4-tiny.weights')
cfg_path = os.path.join(main_path, 'yolov4-tiny.cfg')

# weight_path=os.path.join(main_path,'yolov3.weights')
# cfg_path=os.path.join(main_path,'yolov3.cfg')

model = cv2.dnn.readNetFromDarknet(cfg_path, weight_path)
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

layers = model.getLayerNames()
unconnected = model.getUnconnectedOutLayers()
unconnected = unconnected - 1

out_layer = []
for i in unconnected:
    out_layer.append(layers[i])


def detect_car(frame, show=False):
    frame_blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), swapRB=True)

    model.setInput(frame_blob)

    detection_layers = model.forward(out_layer)

    boxes_list = []
    confidence_list = []
    ids_list_num = []

    h, w = frame.shape[:2]
    for detec_layer in detection_layers:

        for detect_object in detec_layer:

            scores_num = detect_object[5:]
            predicted_id_num = np.argmax(scores_num)
            confidence_num = scores_num[predicted_id_num]

            if confidence_num > .20:
                bounding_box = detect_object[0:4] * np.array([w, h, w, h])

                (box_cen_x, box_cen_y, box_width, box_height) = bounding_box.astype('int')
                start_x = int(box_cen_x - (box_width / 2))
                start_y = int(box_cen_y - (box_height / 2))

                if start_x < 0:
                    start_x = -1 * start_x
                if start_y < 0:
                    start_y = -1 * start_y

                confidence_list.append(float(confidence_num))
                ids_list_num.append(predicted_id_num)
                boxes_list.append([start_x, start_y, int(box_width), int(box_height)])

    max_ids = cv2.dnn.NMSBoxes(boxes_list, confidence_list, .5, .3)

    car_num = 1
    coordinates = []
    for max_id in max_ids:
        box = boxes_list[max_id]

        x0 = box[0]
        y0 = box[1]
        x1 = int(x0 + box[2])
        y1 = int(y0 + box[3])

        coordinates.append([x0, y0, x1, y1])

        label_id_num = ids_list_num[max_id]
        if label_id_num == 2 or label_id_num == 5 or label_id_num == 7:
            if show:
                confidence_score = int(100 * confidence_list[max_id])
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
                cv2.putText(frame, str(confidence_score), (x0, y0 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                cv2.putText(frame, f'detected car num:{len(max_ids)}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, \
                            1, (0, 255, 0), 2)

            car_num = car_num + 1

    return coordinates




import cv2
import numpy as np
import json
import time
from ovmsclient import make_grpc_client
from tabulate import tabulate
import os
import datetime
import threading
import queue

class YOLOv8OVMS:
    def __init__(self, rtsp_url, class_names, input_shape, color_palette, confidence_thres, iou_thres, model_name, ovms_url, save_img_loc, skip_rate, verbose=False):
        print(f"Initializing YOLOv8OVMS with RTSP URL: {rtsp_url}")
        self.rtsp_url = rtsp_url
        self.class_names = class_names
        self.input_width, self.input_height = input_shape
        self.color_palette = np.random.uniform(0, 255, size=(len(class_names), 3))
        self.confidence_thres=confidence_thres
        self.iou_thres=iou_thres
        self.model_name=model_name
        self.ovms_url=ovms_url
        self.save_img_loc=save_img_loc
        self.verbose=verbose
        self.frame_number =0
        self.skip_rate=skip_rate
        self.grpc_client = make_grpc_client(ovms_url)
        self.stopped = False
        self.lock = threading.Lock()
        self.preprocessed_frames_queue = queue.Queue(maxsize=150)
        self.inferenced_frames_queue = queue.Queue(maxsize=150)
        self.postprocessed_frames_queue = queue.Queue(maxsize=150)
        self.capture_thread = threading.Thread(target=self.capture_frames)
        self.capture_thread.start()
        self.postprocess_thread = threading.Thread(target=self.postprocess_frames)
        self.postprocess_thread.start()
        self.inference_thread = threading.Thread(target=self.run_inference)
        self.inference_thread.start()
        self.total_frames = 0
        self.total_fps = 0
        self.start_time = time.time()

    def capture_frames(self):
        cap = cv2.VideoCapture(self.rtsp_url)
        while not self.stopped:
            ret, frame = cap.read()
            if not ret:
                self.log("Failed to grab frame")
                print("Failed to grab frame")
                break
            
            preprocessed_frame = self.preprocess(frame)
            frame_tuple = (frame, preprocessed_frame)
            if self.preprocessed_frames_queue.full():
                time.sleep(0.01)
            print("Adding frame to preprocessed frames queue...")
            self.preprocessed_frames_queue.put(frame_tuple)
        cap.release()

    def postprocess_frames(self):
        while not self.stopped:
            while(self.inferenced_frames_queue.empty()):
                time.sleep(0.01)
            print("Postprocessing frames...")
            frame, outputs = self.inferenced_frames_queue.get()
            
            postprocessed_frame = self.postprocess(frame, outputs)
            while self.postprocessed_frames_queue.full():
                time.sleep(0.01)
            print("Adding postprocessed frame to postprocessed frames queue...")

            self.total_frames += 1
            total_fps = self.total_frames / (time.time() - self.start_time)
            print("total fps is: ", total_fps)
            self.postprocessed_frames_queue.put(postprocessed_frame)        
  
    def preprocess(self, frame):
        if(self.verbose):
            print("Preprocessing the frame...")

        self.img_height, self.img_width = frame.shape[:2]  # Actualiza las dimensiones basadas en el frame actual
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_width, self.input_height))
        image_data = np.array(img) / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        return image_data

    def postprocess(self, input_image, output):
        if(self.verbose):
            print("Postprocessing the output...")

        # Transpose and squeeze the output to match the expected shape
        outputs = np.transpose(np.squeeze(output[0]))

        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []

        # Calculate the scaling factors for the bounding box coordinates
        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height

        # Iterate over each row in the outputs array
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i, 4:]

            # Find the maximum score among the class scores
            max_score = np.max(classes_scores)

            # If the maximum score is above the confidence threshold
            if max_score >= self.confidence_thres:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row
                x, y, w, h = outputs[i, 0:4]

                # Calculate the scaled coordinates of the bounding box
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                # Add the class ID, score, and box coordinates to the respective lists
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)

        # Check if indices are returned as a numpy array and access them correctly
        if len(indices) > 0:
            indices = indices.flatten()  # This ensures indices are flattened properly
        elif(self.verbose):
            print("No boxes to display after NMS.")

        # Prepare data for tabulate
        table_data = []

        # Iterate over the selected indices after non-maximum suppression
        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]

            # Draw the detection on the input image
            self.draw_detections(input_image, box, score, class_id)
            table_data.append([i, box, score, self.class_names[class_id]])

        # Print the table
        headers = ["Index", "Box", "Score", "Class"]
        #self.log(self, str(tabulate(table_data, headers=headers, tablefmt="grid")))
        
        # Return the modified input image
        return input_image

    def draw_detections(self, img, box, score, class_id):
        # Extract the coordinates of the bounding box
        x1, y1, w, h = box

        # Retrieve the color for the class ID
        color = self.color_palette[class_id]

        # Draw the bounding box on the image
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (0,0,255), 5)

        # Create the label text with class name and score
        label = f"{self.class_names[class_id]}: {score:.2f}"

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 1)

        # Calculate the position of the label text
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 20

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(img, (label_x, label_y - label_height - 10), (label_x + label_width, label_y + label_height), (0,0,255), cv2.FILLED)

         # Draw the label text on the image
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 1, cv2.LINE_AA)

    def run_inference(self):
        while not self.stopped:
            while(self.preprocessed_frames_queue.empty()):
                time.sleep(0.01)
            
            frame, image_data = self.preprocessed_frames_queue.get()
            outputs = self.grpc_client.predict({"images": image_data}, self.model_name)

            while(self.inferenced_frames_queue.full()):
                time.sleep(0.01)
            
            print("Adding frame and outputs to inferenced frames queue...")
            frame_tuple = (frame, outputs)
            self.inferenced_frames_queue.put(frame_tuple)   

    def run(self):
        while not self.stopped:
            if(self.verbose):
                print("Running detection...")

            self.frame_number += 1
            # If mod = 0, i will get the frame and skip it
            if self.frame_number % self.skip_rate == 0:
                if not self.frame_queue.empty():
                    self.cap = self.frame_queue.get()
                continue
            
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                # Preprocess the current frame
                image_data = self.preprocess(frame)

                # Send the preprocessed frame to the gRPC client for prediction
                outputs = self.grpc_client.predict({"images": image_data}, self.model_name)

                # Postprocess the prediction results and get the final frame
                processed_frame = self.postprocess(frame, outputs)

                return processed_frame
            else:
                print("Frame queue is empty. Waiting for 10 ms...")
                time.sleep(0.01)

    def stop(self):
        with self.lock:
            self.stopped = True
        self.capture_thread.join()
    
    def log(self, message):
        """Logs a message with a timestamp if verbose is true."""
        if self.verbose:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{timestamp} - {message}")
            
    def __del__(self):
        print("Releasing resources...")
        #self.cap.release()
        self.stop()
        cv2.destroyAllWindows()
        print("Released video capture and destroyed all windows.")
    
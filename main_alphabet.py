import cv2
import mediapipe as mp
import numpy as np
from torch_geometric.data import Data
import torch
from model_alphabet import EnhancedHandPoseGNN



model = EnhancedHandPoseGNN() 

mapping = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


edge_list = [(0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8), (0,9), (9,10), (10,11), (11,12), (0,13), (13,14), (14,15), (15,16), (0,17), (0,18), (0,19), (0,20)] # Example edges

edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
# Assuming you have loaded the state dict from the saved model
state_dict = torch.load('models/model_alphabet_10_epochs.pth')


model.load_state_dict(state_dict)
model.eval()


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 2
color = (0, 255, 0) 
thickness = 3
line_type = cv2.LINE_AA


cap = cv2.VideoCapture(0)
_, frame = cap.read()
h, w, _ = frame.shape

if (cap.isOpened() == False):
    print("error")
    
    
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    
    while cap.isOpened():
        
        ret, frame = cap.read()
        
        if ret:
            
            frame.flags.writeable = False
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
            
            output = hands.process(frame)
            
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            new_data = []
            
            if output.multi_hand_landmarks:
                for landmark in output.multi_hand_landmarks:
                    
                    x_max = 0
                    y_max = 0
                    x_min = w
                    y_min = h
                
                    for coordinate in landmark.landmark:
                        
                        x = coordinate.x
                        y = coordinate.y
                        
                        new_data.append(x)
                        new_data.append(y)
                        
                        x_int = int((x * w))
                        y_int = int((y * h))
                        
                        x_max = max(x_max, x_int)
                        y_max = max(y_max, y_int)
                        
                        x_min = min(x_min, x_int)
                        y_min = min(y_min, y_int)
                        
                        x_margin = int((x_max - x_min) * 0.015)
                        y_margin = int((y_max - y_min) * 0.015)
                        
                        x_max += x_margin
                        y_max += y_margin
                        
                        x_min -= x_margin
                        y_min -= y_margin
                    
        
                        
                    mp_drawing.draw_landmarks(
                        frame,
                        landmark,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    
                    
            final_data = np.array([new_data])       
            
            
            if len(new_data) == 42:  
                
                keypoints = np.array(new_data).reshape(-1, 2) 
                keypoints = (keypoints - keypoints.mean(axis=0)) / keypoints.std(axis=0)
                x = torch.tensor(keypoints, dtype=torch.float)
                
                
                
                input = Data(x=x, edge_index=edge_index)

                with torch.no_grad():

                    output = model(input)
                    _, predicted = torch.max(output.data, 1)
            
                
                final_output = predicted.item()
                
                text = "Prediction: " + mapping[final_output]

                position = (x_min, y_min - 5)
                
                cv2.putText(frame, text, position, font, font_scale, color, thickness, line_type)

                    
            
            cv2.imshow("webcam", frame)
            
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        
        else:
            break
    

cap.release()

cv2.destroyAllWindows()
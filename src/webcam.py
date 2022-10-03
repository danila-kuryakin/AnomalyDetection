from datetime import datetime
import time

import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import numpy as np
import csv, cv2, os, torch
import torch.utils.data as data
import pandas as pd
import mediapipe as mp


class Res18Feature(nn.Module):
    def __init__(self, pretrained=True, num_classes=7):
        super(Res18Feature, self).__init__()
        resnet = models.resnet18(pretrained)
        # self.feature = nn.Sequential(*list(resnet.children())[:-2]) # before avgpool
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # after avgpool 512x1

        fc_in_dim = list(resnet.children())[-1].in_features  # original fc layer's in dimention 512

        self.fc = nn.Linear(fc_in_dim, num_classes)  # new fc layer 512x7
        self.alpha = nn.Sequential(nn.Linear(fc_in_dim, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)

        attention_weights = self.alpha(x)
        out = attention_weights * self.fc(x)
        return attention_weights, out


def run_test():
    model_path = '../models/RAF_basic_acc0.8625.pth'
    # model_path = '../models/epoch17_acc0.625_compound2.pth'
    # model_path = '../models/FER_acc0.6802.pth'

    labels_rafb = {0: 'Surprise',
              1: 'Fear',
              2: 'Disgust',
              3: 'Happiness',
              4: 'Sadness',
              5: 'Anger',
              6: 'Neutral'}

    labels_rafc = {0: 'Happily Surprised',
                1: 'Happily Disgusted',
                2: 'Sadly Fearful',
                3: 'Sadly Angry',
                4: 'Sadly Surprised',
                5: 'Sadly Disgusted',
                6: 'Fearfully Angry',
                7: 'Fearfully Surprised',
                8: 'Angrily Surprised',
                9: 'Angrily Disgusted',
                10: 'Disgustedly Surprised'}


    imagenet_pretrained = True
    res18 = Res18Feature(pretrained=imagenet_pretrained)

    print("Loading pretrained weights...", model_path)
    pretrained = torch.load(model_path)
    pretrained_state_dict = pretrained['model_state_dict']
    model_state_dict = res18.state_dict()
    loaded_keys = 0
    total_keys = 0
    for key in pretrained_state_dict:
        if ((key == 'module.fc.weight') | (key == 'module.fc.bias')):
            pass
        else:
            model_state_dict[key] = pretrained_state_dict[key]
            total_keys += 1
            if key in model_state_dict:
                loaded_keys += 1
    print("Loaded params num:", loaded_keys)
    print("Total params num:", total_keys)
    res18.load_state_dict(model_state_dict, strict=False)

    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    res18 = res18.cuda()

    mp_face_detection = mp.solutions.face_detection
    cap = cv2.VideoCapture(0)
    with mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5) as face_detection:
        while cap.isOpened():
            start_time = time.time()
            success, image = cap.read()

            if not success:
                print("Ignoring empty camera frame.")

            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detection.process(gray_image)

            if results.detections:
                h, w, _ = image.shape
                x1 = int(results.detections[0].location_data.relative_bounding_box.xmin * w)
                y1 = int(results.detections[0].location_data.relative_bounding_box.ymin * h)
                width = int(results.detections[0].location_data.relative_bounding_box.width * w)
                height = int(results.detections[0].location_data.relative_bounding_box.height * h)

                x2 = x1 + width
                y2 = y1 + height
                face = image[y1:y2, x1:x2, :]

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 0)


                with torch.no_grad():
                    res18.eval()
                    transform_image = data_transforms(face)
                    image_4d = transform_image.unsqueeze(0)

                    _, outputs = res18(image_4d.cuda())
                    _, predicts = torch.max(outputs, 1)

                    h, w, _ = image.shape
                    pos = (w - x2, y1)
                    text_label = labels_rafb[predicts.item()]

                    flip_image = cv2.flip(image, 1)

                    draw_text(img=flip_image, text=text_label,
                                font=cv2.FONT_HERSHEY_PLAIN,
                                pos=pos,
                                font_scale=1.3,
                                font_thickness=1,
                                text_color=(255, 255, 255),
                                text_color_bg=(0, 0, 255)
                                )

            fps_label = '%s FPS' % (1//(time.time() - start_time))
            draw_text(img=flip_image, text=fps_label,
                      font=cv2.FONT_HERSHEY_PLAIN,
                      pos=(2, 2),
                      font_scale=1.8,
                      font_thickness=1,
                      text_color=(255, 255, 255),
                      text_color_bg=(0, 0, 255)
                      )
            cv2.imshow('Cap', flip_image)
            if cv2.waitKey(5) & 0xFF == 27:
                break

def draw_text(img, text,
          font=cv2.FONT_HERSHEY_PLAIN,
          pos=(0, 0),
          font_scale=3,
          font_thickness=2,
          text_color=(0, 255, 0),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, int(y + text_h + font_scale - 1)), font, font_scale, text_color, font_thickness)

    return text_size

if __name__ == "__main__":
    # run_training()
    run_test()
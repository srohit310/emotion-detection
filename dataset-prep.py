import numpy as np
import pandas as pd
import cv2
from os import system, name

df = pd.read_csv('affectnet/labels.csv')

emotion_dict = {'anger':0, 'disgust':1, 'fear':2, 'happy':3, 'sad':4, 'surprise':5, 'neutral':6}
dict = {"pixels":[],"emotion":[]}

for index, row in df.iterrows():

    path = row['pth']
    img = cv2.imread('affectnet/'+path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48,48), interpolation=cv2.INTER_AREA)
    img = img.flatten(order='C')
    pixels = ''
    
    for num in img:
        pixels += str(num)+' '
    pixels = pixels.strip()

    emotion = row['label']

    if emotion in emotion_dict:

        dict['pixels'].append(pixels)
        dict['emotion'].append(emotion_dict[emotion])
    
    system('cls')
    print("Done with index "+str(index))

df = pd.DataFrame(dict)
print(df.info())
df.to_csv("affectnet.csv", index=False)

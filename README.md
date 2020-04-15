## Emotion Detection

Developed a program that utilises the user's webcam to detect users facial expressions. This was created using tranfer learning with Google's pretrained model, MobileNet V2. The analysis and modelling conducted on Jupyter Notebook can be viewed [here].

Data: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

[here]: https://github.com/j-truong/Emotion-Detection/blob/master/emotion_detection.ipynb

![Alt Text](https://github.com/j-truong/Emotion-Detection/blob/master/images/webcam_gif.gif)

## Prerequisites

```
pip install requirements.txt
```

## Script
Run webcam program
```
python webcam.py
```

## Results
### Analysis
After preprocessing the Kaggle dataset, I wanted to inspect the data to investigate to what is contained. Plotting a count plot of each emotion provided insight of potential vulnerabilities within my model. With significant difference in data between 'Disgust' and the other emotions, I would expect the model to find difficulties in classifying the 'Disgust' emotion due to insufficient data. Counterintuitively, we should expect the model to classify 'Happy' with greater accuracy as there exists an abundance of data as well as being a distinctive facial expression. To define what is distinctive, I believe emotions such as 'Happy' and 'Angry' are easy to identify by eye as they have noticable traits such as converging eyebrows for 'Angry' or a curved mouth for 'Happy'. 

![image](https://github.com/j-truong/Emotion-Detection/blob/master/images/faces.png)

![image](https://github.com/j-truong/Emotion-Detection/blob/master/images/emotion_count.png)

I 

![image](https://github.com/j-truong/Emotion-Detection/blob/master/images/acc_loss.png)

![image](https://github.com/j-truong/Emotion-Detection/blob/master/images/f1score.png)

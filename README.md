## Emotion Detection

Developed a webcam-based program that utilises CNNs and transfer learning to detect and classify user(s) facial expressions. This was created using with Google's pretrained model, MobileNet V2. The analysis and modelling conducted on Jupyter Notebook can be viewed [here].

https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

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
After preprocessing the Kaggle dataset, I wanted to inspect the data to investigate to what is contained. Plotting a count plot of each emotion provided insight of potential vulnerabilities within my model. With significant difference in data between 'Disgust' and the other emotions, I would expect the model to find difficulties in classifying the 'Disgust' emotion due to insufficient data. Counterintuitively, we should expect the model to classify 'Happy' with greater accuracy as there exists an abundance of data as well as being a distinctive facial expression. To define what is distinctive, I believe emotions such as 'Happy' and 'Angry' are easy to identify by eye and easily seperable to the other emotions as they have noticable traits such as steep, converging eyebrows for 'Angry' or a curved mouth for 'Happy'. 

![image](https://github.com/j-truong/Emotion-Detection/blob/master/images/faces.png)

![image](https://github.com/j-truong/Emotion-Detection/blob/master/images/emotion_count.png)

### Model
Using the Google's pretrained MobileNet V2 model (containing 155 layers), I ran 3 models; one was where the only top was unfrozen(TL top), another with 100 layers frozen (TL 100), and finally, one with the entire model was unfrozen (TL whole). As expected with a decent amount data, the model progressed as more layers unfroze; reaching to a final validation accuracy of 96.9139 % in TL whole.

![image](https://github.com/j-truong/Emotion-Detection/blob/master/images/acc_loss.png)

### Conclusion
Continuinig from the TL whole model, below displays the F1 Score of each emotion with relatively high scores, including 'Disgust'. But to compare this to the practical use of the webcam program, I personally found it difficult to obtain the 'Disgust' and 'Surprise' emotion which may be due to my intuition from earlier (lack of data). For future improvement, this problem could perhaps be solved by applying Data Augmentation to the 'Disgust' and 'Surprise' datasets to gain a better classification on these emotions. 

![image](https://github.com/j-truong/Emotion-Detection/blob/master/images/f1score.png)

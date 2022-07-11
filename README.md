# Face-Emoji-Detection
CNN based and Dlib Face Landmark Predictor for Face Emoji Prediction

## Introduction:

This repository consists of two sections: 
1. CNN based Facial Emoji Prediction 
2. Dlib Landmark Predictor for Face Emoji prediction



##### Overview for Deep Learning for Emojis - 
Nowadays, we are using several emojis or avatars to show our moods or feelings. They act as nonverbal cues of humans. They become the crucial part of emotion recognition, online chatting, brand emotion, product review, and a lot more. Data science research towards emoji-driven storytelling is increasing repeatedly.

The detection of human emotions from images is very trendy, and possibly due to the technical advancements in computer vision and deep learning. In this deep learning project, to filter and locate the respective emojis or avatars, we will classify the human facial expressions. 

# Dataset :
This article was published as a part of the Data Science Blogathon.

Overview for Deep Learning for Emojis
Nowadays, we are using several emojis or avatars to show our moods or feelings. They act as nonverbal cues of humans. They become the crucial part of emotion recognition, online chatting, brand emotion, product review, and a lot more. Data science research towards emoji-driven storytelling is increasing repeatedly.

The detection of human emotions from images is very trendy, and possibly due to the technical advancements in computer vision and deep learning. In this deep learning project, to filter and locate the respective emojis or avatars, we will classify the human facial expressions. If you are not familiar with deep learning you can .

## About the Dataset :
The pictorial dataset we are going to use for this project is FER2013 (Facial Expression Recognition 2013). It contains 48*48-pixel grayscale face images. The images are located in the center and occupy the same amount of space. Below is the facial expression categories present in our dataset:

- 0:angry   - 3995

- 1:disgust - 436

- 2:fear    - 4097

- 3:happy   - 7215

- 4:sad     - 4965

- 5:surprise - 4830

- 6:natural  - 3171

Training Dataset has 28709 images belonging to 7 classes. 
Test Dataset has 7178 images belonging to 7 classes.

<p align = "center">
<img src ="https://github.com/Tanvi-Sharmaa/face-emoji-prediction/blob/main/dataset.png" align = "center"/>
</p>
<br>

# Approach - 
Firstly, we build a deep learning model which classifies the facial expressions from the pictures. Then we will locate the already classified emotion with an avatar or an emoji.

### Model Image - 

<p align = "center">
<img src ="https://github.com/Tanvi-Sharmaa/face-emoji-prediction/blob/main/model.png" align = "center"/>
</p>
<br>

# Dlib Landmark Predictor :

In order to convert face into an emoji in real-time firstly a face need to be identified and it should be isolated from the background. Recognize all the key landmark features from the face. Facial landmark is a subset of the shape prediction problem. Given an input image (and normally an ROI that specifies the object of interest), a shape predictor attempts to localize key points of interest along with the shape.

In the context of facial landmarks, our goal is to detect important facial structures on the face using shape prediction methods.

The entire project is a 3-step process:

Step #1: Identify the face in the image.
Step #2: Detect the key facial structures from the face.
Step #3: Convert facial features to a real-time emoji.

## #1: Identify the face in the image.

Dlib for face detection uses a combination of HOG (Histogram of Oriented Gradient) & Support Vector Machine (SVM) which is trained on positive and negative images (meaning there are images that have faces and ones that don’t).

## #2: Detect the key facial structures from the face:

There are a variety of facial landmark detectors, but all methods essentially try to localize and label the following facial regions:

- Mouth
- Right eyebrow
- Left eyebrow
- Right eye
- Left eye
- Nose
- Jaw

The end result is a facial landmark detector that can be used to detect facial landmarks in real-time with high-quality predictions.

## #3: Convert facial features to a real-time emoji:

In real-time we are finding facial landmarks, now we need to convert the landmark position to the corresponding component in the emoji.

<p align = "center">
<img src ="https://github.com/Tanvi-Sharmaa/face-emoji-prediction/blob/main/emoji_map.jpeg" align = "center"/>
</p>
<br>


# Result :

<p align = "center">
<img src ="https://github.com/Tanvi-Sharmaa/face-emoji-prediction/blob/main/result.png" align = "center"/>
</p>
<br>

# Applications :
In recent years, the use of facial expression recognition systems or softwares have been increasing. There are many applications of this product.
#### Market Research: 
Based on the current customers and partners engagements, the company is able to deduce marketing strategies and customized products in order to retain them. An expression reader in this situation can be useful as it can record the customers’ or partners’ emotions by advertisements or meetings and thus aid the company with this data.
#### Health and Medicine: 
In the field of health, this product can help in many methods. In every aspect of health care, it can be used to determine the emotions of a patient undergoing a surgery or a treatment process and notify the respective doctor about them. Using them as a basis a doctor can advise a better treatment suitable for the patient and thus easing out the difficulties or pain for them.
#### Psychology: 
It is the scientific study of the human mind and its functions, and it is done for the purpose of education or research including the medicinal part. This product has a lot of uses in the field of psychology as psychology is very closely related to emotions. Thus, learning emotions of humans at different situations can give us a much better knowledge about his/her mental psychology.
#### Autism: 
It refers to a broad range of conditions which are characterized by challenges with social skills, repetitive behaviors, speech and nonverbal communication. Thus for a person who is incapable of reading expressions of the society and understanding their emotions, face expression recognition system can aid a lot by characterizing it for such people and thus make it easier for them to react in a social gathering and thereby easing their lives.
#### Lie Detection: 
The machine developed for lie detection uses the sensors to find any abnormality in the blood flow streams of the user. This product can aid it by characterizing any suspicious changes in the expressions/emotions of the person who is being tested.
#### Security System: 
In a environment like an ATM or a bank, where the treat to security is a major problem, the security cameras can be enabled with this software in order to detect any suspicious expression to enhance the precautionary measures.
#### Education: 
This product can be used to measure real-time learner responses and engagement with their educational content and thus adapt, personalize the content and measure effectiveness of the lecturer.

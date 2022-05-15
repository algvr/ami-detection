# Analysis of Electrocardiograms via Artificial Neural Networks For a Reliable Assessment of a Possible Myocardial Infarction
## Instructions for the webtool  

This webtools is designed to work on desktop and mobile browsers. The procedure for processing an ECG diagrams to make predictions is as follows:  

1. Click the "Upload Photo" button and select the image of the ECG diagram wished to be analysed. Preferable, the image should be taken under the best possible conditions on a flat surface.  

2. After the image is loaded, an image editor will appear on the screen, and the leads have to be signalized accordingly by the user. To do so, click on the button "new lead" and adjust position, size, and orientation of the box. Repeat for all leads in the ECG diagram, the final result should look like a well-oriented grid of boxes overlapping the ECG diagram in the image.  

3. When all leads are signalized properly, click the button for "process ECG" and the sections of the image that were signalized to be leads will be sent to the neural network for processing. The results should be sent back in a few moments.

## Disclaimers about this webtool  

### Objectives' statement  

This webtools aims to illustrate the capacity of deep-learning neural networks to learn to classify anomalies in ECG readouts based on human-labeled training data.  

The data to train the network consists of classifications done by a specialized medical professional. The network is trained with the objective of predicting how a medical professional would classify a given ECG diagram, but by no means should be taken as a medical opinion for making decisions regarding health concerns.  

If you think you may have a medical emergency, call your doctor or emergency services immediately.  

### Data protection  

We reserve the right to collect and analyze any information uploaded to this website in accordance with the Objectives' statement of this website. We follow the provisions given in the Federal Act on Data Protection in Switzerland.  

### Exclusion of liability  

All contents within this website and in any other website linked here are not guaranteed to contain accurate, up to date, nor complete information. We are in no way liable, by none of the legal liability reasons, for any direct, indirect or consequential damage to any user of the information presented in this website, or any other website accessed via links here.  

### About the outputs of the webtool  

Notice that the neural network will always output a finite probability for all of the 3 classes. This results should not be taken as accurate professional medical advice.  

If you think you may have a medical emergency, call your doctor or emergency services immediately.  

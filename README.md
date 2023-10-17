#Deep Learning Approach for Detecting Neurons of the Primary Visual Cortex and Orientation Selectivity

  This repository contains code and resources for a novel deep-learning approach for detecting neurons of the primary visual cortex (V1) and determining their orientation selectivity. The approach combines deep learning models and signal processing techniques to automatically detect V1 neurons and analyze their preferred orientation.
  Abstract
  
  The primary goal of this research is to leverage calcium imaging microscopic technique to detect and study the activity of neurons in the primary visual cortex. Calcium imaging involves the use of fluorescent calcium indicators to measure the levels of calcium ions (Ca2+) in cells or tissues. By monitoring changes in Ca2+ levels, it is possible to track the activity of thousands of neurons in the mammalian brain, particularly in the primary visual cortex.
  
  Neurons in the primary visual cortex exhibit orientation selectivity, which refers to their responsiveness to specific visual stimuli with specific orientations. This research focuses on studying the activity of V1 neurons and determining their preferred orientation. Traditional methods for studying orientation selectivity require manual extraction of data for each neuron, which is time-consuming. Additionally, processing calcium imaging data computationally is expensive.
  
  To address these challenges, this research presents a pipeline that combines deep learning models and signal processing techniques. The pipeline begins with the automatic detection of V1 neurons using deep learning models. The resulting average F-1 score achieved on the testing data is 0.89, which outperforms state-of-the-art methods that typically achieve a testing F-1 score below 0.85.
  
  Once the V1 neurons are detected, data is extracted for each neuron, and signal processing techniques are applied to determine their preferred orientation. The orientation selectivity of V1 neurons has played a crucial role in advancing our understanding of vision and has also inspired the development of modern computer vision models such as Convolutional Neural Networks (CNN).

## Video Tutorial

[![YouTube Video](https://img.youtube.com/vi/F103alwgWHA/0.jpg)](https://youtu.be/F103alwgWHA)




## Requirements
```
streamlit_lottie==0.0.2
streamlit==1.3.0
requests==2.24.0
Pillow==8.4.0
```

## Run the app
```
streamlit run app.py
```

## Uploading Videos in ML page
```
- if you want to upload video in ml page,you should upload it in zip format only
- Dont go to Data Analysis page without running ML page
```

## Author
- graduation projects for FCAI -Abdelrahman Hassan-



## Feedback

If you have any feedback, please reach out to me at abdohassan7001@gmail.com



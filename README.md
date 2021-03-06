# GoCV DNN Text Detection

A implementation of the OpenCV `/dnn/text_detection` sample written in Go, and using the GoCV package. The implementation
attempts to stay as faithful as the original C++ and Python sample provided in the OpenCV repository which uses the pre-trained
EAST Text Detection Tensorflow Model.

## Requirements

In order to run the example, the following must be installed:

- GoCV v0.27
- OpenCV 4.5.1

Instructions to install these two dependencies can be found [here](https://gocv.io/getting-started/)

The EAST Text Detection Model and the Text Recognition Model must be downloaded via the links in the Resources section below.

## How to Run

Specify a file in the `input` if you wish to perform text detection against a static image.

```bash
go run main.go -input images/bottle.jpg -model frozen_east_text_detection.pb -ocr CRNN_VGG_BiLSTM_CTC.onnx 
```

Omit the `input` flag if you wish to perform text detection from a camera source. Other optional flags are available
to adjust the default confidence levels.

## Sample Results

When ran against the `bottle.jpg` sample image, the EAST Text Detection Model correctly identifies the text on this bottle.
![image](https://user-images.githubusercontent.com/47725056/118376282-13b02e80-b595-11eb-80a5-4ad9c97d5a91.png)

It is also able to correctly identify the words `ALTO` in this image below.
![image](https://user-images.githubusercontent.com/47725056/118376327-5d991480-b595-11eb-8897-dbec0a79b27e.png)

## Resources

- [OpenCV C++ Sample](https://github.com/opencv/opencv/blob/master/samples/dnn/text_detection.cpp)
- [OpenCV Python Sample](https://github.com/opencv/opencv/blob/master/samples/dnn/text_detection.py)
- [EAST Text Detection Model Download](https://github.com/argman/EAST)
- [Text Recognition Model Download](https://drive.google.com/drive/folders/1cTbQ3nuZG-EKWak6emD_s8_hHXWz7lAr?usp=sharing)

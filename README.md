# GoCV DNN Text Detection

A implementation of the OpenCV `/dnn/text_detection` sample written in Go, and using the GoCV package. The implementation
attempts to stay as faithful as the original C++ and Python sample provided in the OpenCV repository which uses the pre-trained
EAST Text Detection Tensorflow Model.

## Requirements

In order to run the example, the following must be installed:

- GoCV v0.27
- OpenCV 4.5.1

Instructions to install these two dependencies can be found [here](https://gocv.io/getting-started/)

## How to Run

Specify a file in the `input` if you wish to perform text detection against a static image.

```bash
go run main.go -input images/bottle.jpg -model frozen_east_text_detection.pb -ocr CRNN_VGG_BiLSTM_CTC.onnx 
```

Omit the `input` flag if you wish to perform text detection from a camera source. Other optional flags are available
to adjust the default confidence levels.

## Resources

C++ Example: <https://github.com/opencv/opencv/blob/master/samples/dnn/text_detection.cpp>
Python Example: <https://github.com/opencv/opencv/blob/master/samples/dnn/text_detection.py>
EAST Text Detection Model: <https://github.com/argman/EAST>
Text Recognition Model Download: <https://drive.google.com/drive/folders/1cTbQ3nuZG-EKWak6emD_s8_hHXWz7lAr?usp=sharing>
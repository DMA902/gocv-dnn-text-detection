/*
    Text detection model: https://github.com/argman/EAST
    Download link: https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.tar.gz?dl=1

    Text recognition models can be downloaded directly here:
    Download link: https://drive.google.com/drive/folders/1cTbQ3nuZG-EKWak6emD_s8_hHXWz7lAr?usp=sharing
    and https://github.com/opencv/opencv/blob/master/doc/tutorials/dnn/dnn_text_spotting/dnn_text_spotting.markdown

    How to convert from pb to onnx:
    Using classes from here: https://github.com/meijieru/crnn.pytorch/blob/master/models/crnn.py
    import torch
    from models.crnn import CRNN
    model = CRNN(32, 1, 37, 256)
    model.load_state_dict(torch.load('crnn.pth'))
    dummy_input = torch.randn(1, 1, 32, 100)
    torch.onnx.export(model, dummy_input, "crnn.onnx", verbose=True)
    For more information, please refer to https://github.com/opencv/opencv/blob/master/doc/tutorials/dnn/dnn_text_spotting/dnn_text_spotting.markdown
	and https://github.com/opencv/opencv/blob/master/doc/tutorials/dnn/dnn_OCR/dnn_OCR.markdown
*/
package main

import (
	"flag"
	"fmt"
	"image"
	"image/color"
	"math"
	"strings"
	"time"

	"gocv.io/x/gocv"
)

func main() {
	input := flag.String("input", "0", "Path to input image or video file. Skip this argument to capture frames from a camera.")
	model := flag.String("model", "", "Path to a binary .pb file contains trained detector network.")
	ocr := flag.String("ocr", "crnn.onnx", "Path to a binary .pb or .onnx file contains trained recognition network")
	width := flag.Int("width", 320, "Preprocess input image by resizing to a specific width. It should be multiple by 32.")
	height := flag.Int("height", 320, "Preprocess input image by resizing to a specific height. It should be multiple by 32.")
	threshold := flag.Float64("thr", 0.5, "Confidence Threshold")
	nms := flag.Float64("nms", 0.4, "Non-maximum suppression threshold.")

	flag.Parse()

	// Load Network
	detector := gocv.ReadNet(*model, "")
	recognizer := gocv.ReadNet(*ocr, "")

	// Create a new named window
	window := gocv.NewWindow("EAST: An Efficient and Accurate Scene Text Detector")
	defer window.Close()

	outNames := []string{
		"feature_fusion/Conv_7/Sigmoid",
		"feature_fusion/concat_3",
	}

	img := gocv.NewMat()
	defer img.Close()

	// Open a video file or an image file or a camera stream
	cap, err := gocv.OpenVideoCapture(*input)
	if err != nil {
		fmt.Printf("Error opening video capture device: %v\n", *input)
		return
	}

	for {
		// Read frame
		cap.Read(&img)
		if img.Empty() {
			continue
		}

		// Get frame height and width
		imgHeight := img.Size()[0]
		imgWidth := img.Size()[1]
		rW := float64(imgWidth) / float64(*width)
		rH := float64(imgHeight) / float64(*height)

		// Create a 4D blob from frame.
		blob := gocv.BlobFromImage(img, 1.0, image.Pt(int(*width), int(*height)), gocv.NewScalar(123.68, 116.78, 103.94, 0), true, false)
		defer blob.Close()

		var inferenceTime time.Duration

		// Run the detection model
		detector.SetInput(blob, "")
		startTime := time.Now()
		outs := detector.ForwardLayers(outNames)
		inferenceTime += time.Since(startTime)

		// Get scores and geometry
		scores := outs[0]
		geometry := outs[1]

		rotatedBoxes, confidences := decodeBoundingBoxes(scores, geometry, float32(*threshold))

		boxes := []image.Rectangle{}
		for _, rotatedBox := range rotatedBoxes {
			boxes = append(boxes, rotatedBox.BoundingRect)
		}

		// Only Apply NMS when there are at least one box
		indices := make([]int, len(boxes))
		if len(boxes) > 0 {
			gocv.NMSBoxes(boxes, confidences, float32(*threshold), float32(*nms), indices)
		}

		// Resize indices to only include those that have values other than zero
		var numIndices int = 0
		for _, value := range indices {
			if value != 0 {
				numIndices++
			}
		}
		indices = indices[0:numIndices]

		for i := 0; i < len(indices); i++ {
			// get 4 corners of the rotated rect
			verticesMat := gocv.NewMat()
			gocv.BoxPoints(rotatedBoxes[indices[i]], &verticesMat)

			// scale the bounding box coordinates based on the respective ratios
			vertices := []image.Point{}
			for j := 0; j < 4; j++ {
				p1 := image.Pt(
					int(verticesMat.GetFloatAt(j, 0)*float32(rW)),
					int(verticesMat.GetFloatAt(j, 1)*float32(rH)),
				)

				p2 := image.Pt(
					int(verticesMat.GetFloatAt((j+1)%4, 0)*float32(rW)),
					int(verticesMat.GetFloatAt((j+1)%4, 1)*float32(rH)),
				)

				vertices = append(vertices, p1)
				gocv.Line(&img, p1, p2, color.RGBA{0, 255, 0, 0}, 1)
			}

			// get cropped image using perspective transform
			if *ocr != "" {
				cropped := fourPointsTransform(img, gocv.NewPointVectorFromPoints(vertices))
				gocv.CvtColor(cropped, &cropped, gocv.ColorBGRToGray)

				// Create a 4D blob from cropped image
				blob := gocv.BlobFromImage(cropped, 1/127.5, image.Pt(100, 32), gocv.NewScalar(127.5, 0, 0, 0), false, false)
				recognizer.SetInput(blob, "")

				// Run the recognition model
				startTime = time.Now()
				result := recognizer.Forward("")
				inferenceTime += time.Since(startTime)

				// decode the result into text
				wordRecognized := decodeText(result)
				gocv.PutText(&img, wordRecognized, vertices[1], gocv.FontHersheySimplex, 0.5, color.RGBA{0, 0, 255, 0}, 1)
			}
		}

		// Put efficiency information
		label := fmt.Sprintf("Inference time: %v ms", inferenceTime.Milliseconds())
		gocv.PutText(&img, label, image.Pt(0, 15), gocv.FontHersheySimplex, 0.5, color.RGBA{0, 255, 0, 0}, 1)

		window.IMShow(img)
		if window.WaitKey(1) >= 0 {
			break
		}
	}
}

func decodeText(scores gocv.Mat) string {
	text := ""
	alphabet := "0123456789abcdefghijklmnopqrstuvwxyz"

	for i := 0; i < scores.Size()[0]; i++ {
		scoresChannel := gocv.GetBlobChannel(scores, 0, i)
		var c int = 0
		var cScore float32 = 0
		for j := 0; j < scores.Size()[2]; j++ {
			score := scoresChannel.GetFloatAt(0, j)
			if cScore < score {
				c = j
				cScore = score
			}
		}

		if c != 0 {
			text += string(alphabet[c-1])
		} else {
			text += "-"
		}
	}

	// adjacent same letters as well as background text must be removed to get the final output
	var charList strings.Builder
	for i := 0; i < len(text); i++ {
		if string(text[i]) != "-" && !(i > 0 && text[i] == text[i-1]) {
			charList.WriteByte(text[i])
		}
	}

	return charList.String()
}

func fourPointsTransform(frame gocv.Mat, vertices gocv.PointVector) gocv.Mat {
	outputSize := image.Pt(100, 32)
	targetVertices := gocv.NewPointVectorFromPoints([]image.Point{
		image.Pt(0, outputSize.Y-1),
		image.Pt(0, 0),
		image.Pt(outputSize.X-1, 0),
		image.Pt(outputSize.X-1, outputSize.Y-1),
	})

	result := gocv.NewMat()
	rotationMatrix := gocv.GetPerspectiveTransform(vertices, targetVertices)
	gocv.WarpPerspective(frame, &result, rotationMatrix, outputSize)

	return result
}

func decodeBoundingBoxes(scores gocv.Mat, geometry gocv.Mat, threshold float32) (detections []gocv.RotatedRect, confidences []float32) {
	scoresChannel := gocv.GetBlobChannel(scores, 0, 0)
	x0DataChannel := gocv.GetBlobChannel(geometry, 0, 0)
	x1DataChannel := gocv.GetBlobChannel(geometry, 0, 1)
	x2DataChannel := gocv.GetBlobChannel(geometry, 0, 2)
	x3DataChannel := gocv.GetBlobChannel(geometry, 0, 3)
	angleChannel := gocv.GetBlobChannel(geometry, 0, 4)

	for y := 0; y < scoresChannel.Rows(); y++ {
		for x := 0; x < scoresChannel.Cols(); x++ {

			// Extract data from scores
			score := scoresChannel.GetFloatAt(y, x)

			// If score is lower than threshold score, move to next x
			if score < threshold {
				continue
			}

			x0Data := x0DataChannel.GetFloatAt(y, x)
			x1Data := x1DataChannel.GetFloatAt(y, x)
			x2Data := x2DataChannel.GetFloatAt(y, x)
			x3Data := x3DataChannel.GetFloatAt(y, x)
			angle := angleChannel.GetFloatAt(y, x)

			// Calculate offset
			// Multiple by 4 because feature maps are 4 time less than input image.
			offsetX := x * 4.0
			offsetY := y * 4.0

			// Calculate cos and sin of angle
			cosA := math.Cos(float64(angle))
			sinA := math.Sin(float64(angle))

			h := x0Data + x2Data
			w := x1Data + x3Data

			// Calculate offset
			offset := []float64{
				(float64(offsetX) + cosA*float64(x1Data) + sinA*float64(x2Data)),
				(float64(offsetY) - sinA*float64(x1Data) + cosA*float64(x2Data)),
			}

			// Find points for rectangle
			p1 := []float64{
				(-sinA*float64(h) + offset[0]),
				(-cosA*float64(h) + offset[1]),
			}
			p3 := []float64{
				(-cosA*float64(w) + offset[0]),
				(sinA*float64(w) + offset[1]),
			}

			center := image.Pt(
				int(0.5*(p1[0]+p3[0])),
				int(0.5*(p1[1]+p3[1])),
			)

			detections = append(detections, gocv.RotatedRect{
				Points: []image.Point{
					{int(p1[0]), int(p1[1])},
					{int(p3[0]), int(p3[1])},
				},
				BoundingRect: image.Rect(
					int(p1[0]), int(p1[1]),
					int(p3[0]), int(p3[1]),
				),
				Center: center,
				Width:  int(w),
				Height: int(h),
				Angle:  float64(-1 * angle * 180 / math.Pi),
			})
			confidences = append(confidences, score)
		}
	}

	// Return detections and confidences
	return
}

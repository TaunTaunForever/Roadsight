# iOS Deployment Guide

This guide explains how to take the current best RoadSight detector and run it on live camera frames on an iPhone or iPad.

## Current Model

The best validated checkpoint in the repo is:

- [best.pt](/home/daniel/Development/Roadsight/runs/detect/runs/train/roadsight_bdd100k_full_yolov8s6_continue10/weights/best.pt)

The exported Apple-ready artifact is:

- [best.mlpackage](/home/daniel/Development/Roadsight/runs/detect/runs/train/roadsight_bdd100k_full_yolov8s6_continue10/weights/best.mlpackage)

This model was trained on the RoadSight class set:

- `car`
- `person`
- `traffic light`
- `traffic sign`
- `bike`

## Deployment Shape

The mobile inference stack is:

1. `AVFoundation` captures frames from the rear camera.
2. `Vision` wraps the Core ML model.
3. The model predicts object detections for each frame.
4. SwiftUI draws boxes and labels over the live preview.

The scaffold files for this flow live in [ios/RoadSightMobile](/home/daniel/Development/Roadsight/ios/RoadSightMobile).

## Create the iOS App

1. Open Xcode and create a new iOS App project.
2. Name it `RoadSightMobile`.
3. Choose `SwiftUI` for the interface and `Swift` for the language.
4. Set an iPhone or iPad deployment target that supports your device.

This repo does not include a full `.xcodeproj` because that tends to be noisy in source control for an otherwise Python-first project. Instead, copy the scaffold source files into your new app target.

## Add the Model

1. Drag [best.mlpackage](/home/daniel/Development/Roadsight/runs/detect/runs/train/roadsight_bdd100k_full_yolov8s6_continue10/weights/best.mlpackage) into the Xcode project navigator.
2. Make sure `Copy items if needed` is enabled.
3. Make sure the app target is checked.

Xcode will compile the package into `best.mlmodelc` inside the app bundle. The scaffold loads that compiled resource directly, so it does not depend on a generated Swift model class.

## Add the Swift Files

Copy these files into the app target:

- [RoadSightMobileApp.swift](/home/daniel/Development/Roadsight/ios/RoadSightMobile/RoadSightMobile/RoadSightMobileApp.swift)
- [ContentView.swift](/home/daniel/Development/Roadsight/ios/RoadSightMobile/RoadSightMobile/ContentView.swift)
- [CameraManager.swift](/home/daniel/Development/Roadsight/ios/RoadSightMobile/RoadSightMobile/CameraManager.swift)
- [DetectorService.swift](/home/daniel/Development/Roadsight/ios/RoadSightMobile/RoadSightMobile/DetectorService.swift)
- [Detection.swift](/home/daniel/Development/Roadsight/ios/RoadSightMobile/RoadSightMobile/Detection.swift)
- [DetectionOverlayView.swift](/home/daniel/Development/Roadsight/ios/RoadSightMobile/RoadSightMobile/DetectionOverlayView.swift)
- [CameraPreviewView.swift](/home/daniel/Development/Roadsight/ios/RoadSightMobile/RoadSightMobile/CameraPreviewView.swift)

## Add Camera Permission

In the target `Info` settings, add:

- `Privacy - Camera Usage Description`

Example value:

`RoadSight uses the camera to run live road-scene object detection.`

Without this, the app will not be allowed to access the camera.

## How the Scaffold Works

### Camera Capture

[CameraManager.swift](/home/daniel/Development/Roadsight/ios/RoadSightMobile/RoadSightMobile/CameraManager.swift) creates an `AVCaptureSession`, attaches the back camera, and emits pixel buffers from `AVCaptureVideoDataOutput`.

### Model Loading

[DetectorService.swift](/home/daniel/Development/Roadsight/ios/RoadSightMobile/RoadSightMobile/DetectorService.swift) looks for `best.mlmodelc` in the app bundle and wraps it in `VNCoreMLModel`.

### Live Inference

Each camera frame is sent through a `VNCoreMLRequest`. The current scaffold uses a simple one-frame-at-a-time guard so the app does not pile up inference requests if the camera is faster than the model.

### Rendering

[DetectionOverlayView.swift](/home/daniel/Development/Roadsight/ios/RoadSightMobile/RoadSightMobile/DetectionOverlayView.swift) converts Vision's normalized bounding boxes into screen coordinates and draws them on top of the preview.

## Running on Device

1. Connect your iPhone or iPad.
2. Select the real device in Xcode.
3. Build and run.
4. Accept camera permission when prompted.

You should then see the live camera feed with RoadSight detections drawn over it.

## Performance Notes

The current exported model is a strong starting point, but on-device performance will depend on the specific Apple chip.

If you need more FPS later, the most practical levers are:

- lower the effective input resolution
- skip frames instead of running on every frame
- raise confidence thresholds so fewer boxes are drawn
- export a lighter checkpoint if needed

## Expected Behavior

The model is best at road-scene content similar to BDD100K:

- cars should be the strongest class
- traffic signs and traffic lights should work reasonably well
- person is moderate
- bike is the weakest class

If you point the camera at indoor scenes or non-driving footage, sparse detections are expected.

## Next Mobile Steps

If you want to push this further after the first live demo works, the best next steps are:

- add a class color palette instead of one box color
- add confidence threshold controls in the UI
- support annotated video recording
- benchmark FPS on a target iPhone or iPad
- add a still-image import flow for saved camera-roll videos and photos

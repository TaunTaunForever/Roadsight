# RoadSightMobile

This folder contains a lightweight iOS scaffold for running the exported RoadSight Core ML model on live camera frames.

It is intentionally not a full Xcode project. The goal is to give you a clean set of Swift files you can drop into a new iPhone or iPad app target while keeping the repo simple.

## Expected Model Artifact

The current recommended on-device model is:

- [best.mlpackage](/home/daniel/Development/Roadsight/runs/detect/runs/train/roadsight_bdd100k_full_yolov8s6_continue10/weights/best.mlpackage)

## Suggested Xcode Setup

1. Create a new iOS App project in Xcode named `RoadSightMobile`.
2. Drag the Swift files from [ios/RoadSightMobile/RoadSightMobile](/home/daniel/Development/Roadsight/ios/RoadSightMobile/RoadSightMobile) into the app target.
3. Drag `best.mlpackage` into the same target.
4. Add `Privacy - Camera Usage Description` to the app `Info.plist`.
5. Run on a real iPhone or iPad, not just the simulator.

## Files

- [RoadSightMobileApp.swift](/home/daniel/Development/Roadsight/ios/RoadSightMobile/RoadSightMobile/RoadSightMobileApp.swift)
- [ContentView.swift](/home/daniel/Development/Roadsight/ios/RoadSightMobile/RoadSightMobile/ContentView.swift)
- [CameraManager.swift](/home/daniel/Development/Roadsight/ios/RoadSightMobile/RoadSightMobile/CameraManager.swift)
- [DetectorService.swift](/home/daniel/Development/Roadsight/ios/RoadSightMobile/RoadSightMobile/DetectorService.swift)
- [Detection.swift](/home/daniel/Development/Roadsight/ios/RoadSightMobile/RoadSightMobile/Detection.swift)
- [DetectionOverlayView.swift](/home/daniel/Development/Roadsight/ios/RoadSightMobile/RoadSightMobile/DetectionOverlayView.swift)
- [CameraPreviewView.swift](/home/daniel/Development/Roadsight/ios/RoadSightMobile/RoadSightMobile/CameraPreviewView.swift)

See [docs/ios_deployment.md](/home/daniel/Development/Roadsight/docs/ios_deployment.md) for the full integration guide.

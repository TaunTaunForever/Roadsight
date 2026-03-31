import AVFoundation
import SwiftUI

struct ContentView: View {
    @StateObject private var cameraManager = CameraManager()
    @StateObject private var detectorService = DetectorService()
    @State private var cameraAuthorized = false

    var body: some View {
        ZStack(alignment: .topLeading) {
            if cameraAuthorized {
                CameraPreviewView(session: cameraManager.session)
                    .ignoresSafeArea()

                DetectionOverlayView(detections: detectorService.detections)
                    .ignoresSafeArea()
            } else {
                Color.black.ignoresSafeArea()
                Text("Camera permission required")
                    .foregroundColor(.white)
            }

            VStack(alignment: .leading, spacing: 8) {
                Text("RoadSight Live")
                    .font(.headline)
                Text(detectorService.statusText)
                    .font(.subheadline)
                Text("Classes: car, person, traffic light, traffic sign, bike")
                    .font(.caption)
            }
            .padding(12)
            .background(.black.opacity(0.65))
            .foregroundColor(.white)
            .clipShape(RoundedRectangle(cornerRadius: 12))
            .padding()
        }
        .task {
            await requestCameraAccess()
            cameraManager.onFrame = { pixelBuffer in
                detectorService.processFrame(pixelBuffer)
            }
            if cameraAuthorized {
                cameraManager.start()
            }
        }
        .onDisappear {
            cameraManager.stop()
        }
    }

    private func requestCameraAccess() async {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            cameraAuthorized = true
        case .notDetermined:
            cameraAuthorized = await AVCaptureDevice.requestAccess(for: .video)
        default:
            cameraAuthorized = false
        }
    }
}

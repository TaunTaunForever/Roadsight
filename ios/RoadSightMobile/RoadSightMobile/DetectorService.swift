import CoreML
import CoreVideo
import Foundation
import Vision

@MainActor
final class DetectorService: ObservableObject {
    @Published var detections: [Detection] = []
    @Published var statusText = "Loading model..."

    private var model: VNCoreMLModel?
    private var isProcessing = false

    init() {
        loadModel()
    }

    func processFrame(_ pixelBuffer: CVPixelBuffer) {
        guard let model, !isProcessing else { return }
        isProcessing = true

        let request = VNCoreMLRequest(model: model) { [weak self] request, error in
            Task { @MainActor in
                defer { self?.isProcessing = false }

                if let error {
                    self?.statusText = "Inference error: \(error.localizedDescription)"
                    return
                }

                let observations = (request.results as? [VNRecognizedObjectObservation]) ?? []
                self?.detections = observations.flatMap(Self.makeDetection(from:))
                self?.statusText = observations.isEmpty ? "No detections" : "\(observations.count) detections"
            }
        }
        request.imageCropAndScaleOption = .scaleFill

        DispatchQueue.global(qos: .userInitiated).async {
            let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right)
            do {
                try handler.perform([request])
            } catch {
                Task { @MainActor in
                    self.statusText = "Vision request failed: \(error.localizedDescription)"
                    self.isProcessing = false
                }
            }
        }
    }

    private func loadModel() {
        do {
            let bundle = Bundle.main
            guard let compiledURL = bundle.url(forResource: "best", withExtension: "mlmodelc") else {
                statusText = "Add best.mlpackage to the app target."
                return
            }

            let coreMLModel = try MLModel(contentsOf: compiledURL)
            model = try VNCoreMLModel(for: coreMLModel)
            statusText = "Model ready"
        } catch {
            statusText = "Model load failed: \(error.localizedDescription)"
        }
    }

    private static func makeDetection(from observation: VNRecognizedObjectObservation) -> [Detection] {
        guard let topLabel = observation.labels.first else { return [] }
        return [
            Detection(
                label: topLabel.identifier,
                confidence: topLabel.confidence,
                boundingBox: observation.boundingBox
            )
        ]
    }
}

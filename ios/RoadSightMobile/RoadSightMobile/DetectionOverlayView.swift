import SwiftUI

struct DetectionOverlayView: View {
    let detections: [Detection]

    var body: some View {
        GeometryReader { proxy in
            ZStack {
                ForEach(detections) { detection in
                    let rect = denormalizedRect(
                        detection.boundingBox,
                        width: proxy.size.width,
                        height: proxy.size.height
                    )

                    ZStack(alignment: .topLeading) {
                        Rectangle()
                            .stroke(Color.green, lineWidth: 2)
                            .frame(width: rect.width, height: rect.height)
                            .position(x: rect.midX, y: rect.midY)

                        Text("\(detection.label) \(Int(detection.confidence * 100))%")
                            .font(.caption.bold())
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(Color.black.opacity(0.7))
                            .foregroundColor(.white)
                            .position(x: rect.minX + 70, y: max(14, rect.minY + 10))
                    }
                }
            }
        }
        .allowsHitTesting(false)
    }

    private func denormalizedRect(_ rect: CGRect, width: CGFloat, height: CGFloat) -> CGRect {
        let x = rect.minX * width
        let y = (1 - rect.maxY) * height
        let w = rect.width * width
        let h = rect.height * height
        return CGRect(x: x, y: y, width: w, height: h)
    }
}

import XCTest
@testable import JTrackers

final class BotSortTests: XCTestCase {
    func testUpdateAssignsTrackId() throws {
        let tracker = BotSort()
        let detections = [
            TrackedObject(x: 10, y: 20, width: 100, height: 200, prob: 0.9),
        ]

        let tracked = try tracker.update(detections).get()

        XCTAssertEqual(tracked.count, 1)
        XCTAssertEqual(tracked.first?.trackId, 1)
        XCTAssertEqual(tracker.frameCount, 1)
    }

    func testUpdateWithReIdFeatures() throws {
        let tracker = BotSort(useReId: true)
        let detections = [
            TrackedObject(x: 10, y: 20, width: 100, height: 200, prob: 0.9),
        ]

        let tracked = try tracker.update(detections, features: [[1, 0, 0]]).get()

        XCTAssertEqual(tracked.count, 1)
        XCTAssertEqual(tracked.first?.trackId, 1)
    }

    func testInvalidFeatureShapeFailsBeforeFfi() {
        let tracker = BotSort(useReId: true)
        let detections = [
            TrackedObject(x: 10, y: 20, width: 100, height: 200, prob: 0.9),
        ]

        let result = tracker.update(detections, features: [[1, 0], [0, 1]])

        XCTAssertEqual(result, .failure(.invalidArgument))
    }
}

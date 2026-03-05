import XCTest
@testable import JTrackers

final class OCSortTests: XCTestCase {
    static func sampleObjects() -> [TrackedObject] {
        [
            TrackedObject(x: 10, y: 20, width: 100, height: 200, prob: 0.9),
            TrackedObject(x: 300, y: 400, width: 80, height: 160, prob: 0.8),
        ]
    }

    func testCreateAndDestroy() {
        let _ = OCSort()
    }

    func testSingleFrameUpdate() throws {
        let tracker = OCSort()
        let results = try tracker.update(Self.sampleObjects()).get()
        XCTAssertGreaterThan(results.count, 0)
    }

    func testMultiFrameTrackId() throws {
        let tracker = OCSort(detThresh: 0.3)
        let objects = Self.sampleObjects()

        var lastResults: [TrackedObject] = []
        for _ in 0..<5 {
            lastResults = try tracker.update(objects).get()
        }

        let ids = lastResults.compactMap(\.trackId)
        XCTAssertFalse(ids.isEmpty, "Expected at least one track ID after multiple frames")
    }

    func testEmptyInput() throws {
        let tracker = OCSort()
        let results = try tracker.update([]).get()
        XCTAssertTrue(results.isEmpty)
    }

    func testCreateWithConfig() throws {
        let tracker = OCSort(
            detThresh: 0.6,
            maxAge: 20,
            minHits: 2,
            iouThreshold: 0.4,
            deltaT: 5,
            inertia: 0.3,
            useByte: true
        )
        let results = try tracker.update(Self.sampleObjects()).get()
        XCTAssertGreaterThan(results.count, 0)
    }

    func testFrameCountAndTrackerCount() throws {
        let tracker = OCSort()
        XCTAssertEqual(tracker.frameCount, 0)
        XCTAssertEqual(tracker.trackerCount, 0)

        _ = try tracker.update(Self.sampleObjects()).get()
        XCTAssertEqual(tracker.frameCount, 1)
        XCTAssertGreaterThan(tracker.trackerCount, 0)
    }
}

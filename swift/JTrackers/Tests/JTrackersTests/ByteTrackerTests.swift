import XCTest
@testable import JTrackers

final class ByteTrackerTests: XCTestCase {
    static func sampleObjects() -> [TrackedObject] {
        [
            TrackedObject(x: 10, y: 20, width: 100, height: 200, prob: 0.9),
            TrackedObject(x: 300, y: 400, width: 80, height: 160, prob: 0.8),
        ]
    }

    func testCreateAndDestroy() throws {
        let _ = try ByteTracker()
    }

    func testSingleFrameUpdate() throws {
        let tracker = try ByteTracker()
        let results = try tracker.update(Self.sampleObjects())
        XCTAssertGreaterThan(results.count, 0)
    }

    func testMultiFrameTrackId() throws {
        let tracker = try ByteTracker(trackThresh: 0.3, highThresh: 0.4, matchThresh: 0.8)
        let objects = Self.sampleObjects()

        var lastResults: [TrackedObject] = []
        for _ in 0..<5 {
            lastResults = try tracker.update(objects)
        }

        let ids = lastResults.compactMap(\.trackId)
        XCTAssertFalse(ids.isEmpty, "Expected at least one track ID after multiple frames")
    }

    func testEmptyInput() throws {
        let tracker = try ByteTracker()
        let results = try tracker.update([])
        XCTAssertTrue(results.isEmpty)
    }
}

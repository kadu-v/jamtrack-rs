import XCTest
@testable import JTrackers

final class FastTrackerTests: XCTestCase {
    func testUpdateAssignsAndPreservesTrackId() throws {
        let tracker = FastTracker()
        let first = try tracker.update([
            TrackedObject(x: 10, y: 20, width: 30, height: 40, prob: 0.9),
        ]).get()
        let second = try tracker.update([
            TrackedObject(x: 11, y: 21, width: 30, height: 40, prob: 0.9),
        ]).get()

        XCTAssertEqual(first.first?.trackId, 1)
        XCTAssertEqual(second.first?.trackId, 1)
        XCTAssertEqual(tracker.frameCount, 2)
        XCTAssertEqual(tracker.trackerCount, 1)
    }

    func testCustomOcclusionAndRoiConfiguration() throws {
        let roi = FastTrackerRoi(
            e1: FastTrackerPoint(x: 0, y: 0),
            e2: FastTrackerPoint(x: 0, y: 100),
            o2: FastTrackerPoint(x: 200, y: 100),
            o1: FastTrackerPoint(x: 200, y: 0)
        )
        let tracker = FastTracker(
            resetVelocityOffset: 10,
            resetPositionOffset: 4,
            enlargeBbox: 1.1,
            dampenMotion: 0.89,
            activeOcclusionToLost: 20,
            initIouSuppression: 0.75,
            rois: [roi],
            roiRepairMaxGap: 8,
            directionWindow: 4,
            directionMarginDegrees: 3
        )
        let output = try tracker.update([
            TrackedObject(x: 10, y: 20, width: 30, height: 40, prob: 0.9),
        ]).get()
        XCTAssertEqual(output.first?.trackId, 1)
    }

    func testEmptyInput() throws {
        let tracker = FastTracker()
        XCTAssertTrue(try tracker.update([]).get().isEmpty)
    }
}

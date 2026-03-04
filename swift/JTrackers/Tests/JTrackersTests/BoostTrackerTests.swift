import XCTest
@testable import JTrackers

final class BoostTrackerTests: XCTestCase {
    static func sampleObjects() -> [TrackedObject] {
        [
            TrackedObject(x: 10, y: 20, width: 100, height: 200, prob: 0.9),
            TrackedObject(x: 300, y: 400, width: 80, height: 160, prob: 0.8),
        ]
    }

    func testCreateAndDestroy() throws {
        let _ = try BoostTracker()
    }

    func testSingleFrameUpdate() throws {
        let tracker = try BoostTracker()
        let results = try tracker.update(Self.sampleObjects())
        XCTAssertGreaterThan(results.count, 0)
    }

    func testMultiFrameTrackId() throws {
        let tracker = try BoostTracker(detThresh: 0.3, iouThreshold: 0.3, maxAge: 30, minHits: 1)
        let objects = Self.sampleObjects()

        var lastResults: [TrackedObject] = []
        for _ in 0..<5 {
            lastResults = try tracker.update(objects)
        }

        let ids = lastResults.compactMap(\.trackId)
        XCTAssertFalse(ids.isEmpty, "Expected at least one track ID after multiple frames")
    }

    func testEmptyInput() throws {
        let tracker = try BoostTracker()
        let results = try tracker.update([])
        XCTAssertTrue(results.isEmpty)
    }

    func testCreateWithConfig() throws {
        let tracker = try BoostTracker(
            detThresh: 0.6,
            iouThreshold: 0.4,
            maxAge: 20,
            minHits: 2,
            lambdaIou: 0.4,
            lambdaMhd: 0.3,
            lambdaShape: 0.3,
            useDloBoost: true,
            useDuoBoost: false,
            enableBoostPlus: true,
            enableBoostPlusPlus: false,
            useShapeSimilarityV1: true
        )
        let results = try tracker.update(Self.sampleObjects())
        XCTAssertGreaterThan(results.count, 0)
    }

    func testCreateWithBoostPlusPlus() throws {
        let tracker = try BoostTracker(
            enableBoostPlusPlus: true
        )
        let results = try tracker.update(Self.sampleObjects())
        XCTAssertGreaterThan(results.count, 0)
    }

    func testFrameCountAndTrackerCount() throws {
        let tracker = try BoostTracker()
        XCTAssertEqual(try tracker.frameCount, 0)
        XCTAssertEqual(try tracker.trackerCount, 0)

        _ = try tracker.update(Self.sampleObjects())
        XCTAssertEqual(try tracker.frameCount, 1)
        XCTAssertGreaterThan(try tracker.trackerCount, 0)
    }

    func testNegativeMaxAgeThrows() throws {
        XCTAssertThrowsError(try BoostTracker(maxAge: -1)) { error in
            XCTAssertEqual(error as? JamTrackError, .invalidArgument)
        }
    }

    func testNegativeMinHitsThrows() throws {
        XCTAssertThrowsError(try BoostTracker(minHits: -1)) { error in
            XCTAssertEqual(error as? JamTrackError, .invalidArgument)
        }
    }
}

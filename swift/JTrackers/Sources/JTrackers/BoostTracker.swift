import JamTrack

/// A multi-object tracker based on the BoostTrack algorithm.
///
/// BoostTrack extends IoU-based tracking with Mahalanobis distance, shape
/// similarity, and confidence-aware association. Optional BoostTrack+ and
/// BoostTrack++ modes enable soft-BIoU and varying-threshold strategies.
///
/// ```swift
/// let tracker = BoostTracker()
/// let detections = [
///     TrackedObject(x: 10, y: 20, width: 100, height: 200, prob: 0.9),
/// ]
/// switch tracker.update(detections) {
/// case .success(let tracked):
///     for obj in tracked {
///         print("Track \(obj.trackId ?? -1)")
///     }
/// case .failure(let error):
///     print("Error: \(error)")
/// }
/// ```
public final class BoostTracker: @unchecked Sendable {
    private let handle: UnsafeMutableRawPointer

    /// Creates a new BoostTracker.
    ///
    /// Negative values for `maxAge` and `minHits` are clipped to `0`.
    ///
    /// - Parameters:
    ///   - detThresh: The detection confidence threshold. Defaults to `0.5`.
    ///   - iouThreshold: The IoU threshold for matching. Defaults to `0.3`.
    ///   - maxAge: The maximum number of frames a track can be lost before removal. Defaults to `30`.
    ///   - minHits: The minimum number of hits before a track is reported. Defaults to `3`.
    ///   - lambdaIou: The weight for IoU cost. Defaults to `0.5`.
    ///   - lambdaMhd: The weight for Mahalanobis distance cost. Defaults to `0.25`.
    ///   - lambdaShape: The weight for shape similarity cost. Defaults to `0.25`.
    ///   - useDloBoost: Whether to use DLO boost. Defaults to `true`.
    ///   - useDuoBoost: Whether to use DUO boost. Defaults to `true`.
    ///   - enableBoostPlus: Whether to enable BoostTrack+ mode. Defaults to `false`.
    ///   - enableBoostPlusPlus: Whether to enable BoostTrack++ mode (takes priority over BoostTrack+). Defaults to `false`.
    ///   - useShapeSimilarityV1: Whether to use shape similarity v1. Defaults to `false`.
    public init(
        detThresh: Float = 0.5,
        iouThreshold: Float = 0.3,
        maxAge: Int = 30,
        minHits: Int = 3,
        lambdaIou: Float = 0.5,
        lambdaMhd: Float = 0.25,
        lambdaShape: Float = 0.25,
        useDloBoost: Bool = true,
        useDuoBoost: Bool = true,
        enableBoostPlus: Bool = false,
        enableBoostPlusPlus: Bool = false,
        useShapeSimilarityV1: Bool = false
    ) {
        let h = jamtrack_boost_tracker_create_with_config(
            detThresh,
            iouThreshold,
            max(maxAge, 0),
            max(minHits, 0),
            lambdaIou,
            lambdaMhd,
            lambdaShape,
            useDloBoost,
            useDuoBoost,
            enableBoostPlus,
            enableBoostPlusPlus,
            useShapeSimilarityV1
        )
        precondition(h != nil, "Failed to create BoostTracker")
        self.handle = h!
    }

    deinit {
        jamtrack_boost_tracker_drop(handle)
    }

    /// Updates the tracker with a new set of detections and returns tracked objects.
    ///
    /// - Parameter objects: An array of ``TrackedObject`` representing the current frame's detections.
    ///   The `trackId` field is ignored on input.
    /// - Returns: A `Result` containing tracked objects with assigned `trackId`, or a ``JamTrackError``.
    public func update(_ objects: [TrackedObject]) -> Result<[TrackedObject], JamTrackError> {
        let cObjects = toCObjects(objects)
        var out = CObjectArray(data: nil, length: 0, _priv: nil)

        let status = cObjects.withUnsafeBufferPointer { buf in
            jamtrack_boost_tracker_update(handle, buf.baseAddress, buf.count, &out)
        }
        defer { jamtrack_object_array_drop(&out) }

        if let error = statusToError(status) {
            return .failure(error)
        }
        return .success(fromCObjectArray(out))
    }

    /// The number of frames processed so far.
    public var frameCount: Int {
        var value: Int = 0
        let status = jamtrack_boost_tracker_frame_count(handle, &value)
        precondition(status == JAMTRACK_STATUS_OK, "Unexpected FFI error: \(status)")
        return value
    }

    /// The number of active tracks.
    public var trackerCount: Int {
        var value: Int = 0
        let status = jamtrack_boost_tracker_tracker_count(handle, &value)
        precondition(status == JAMTRACK_STATUS_OK, "Unexpected FFI error: \(status)")
        return value
    }
}

import JamTrack

/// A multi-object tracker based on the BoT-SORT algorithm.
///
/// BoT-SORT combines BYTE-style two-stage association with an `xywh` Kalman
/// filter, optional camera-motion compensation, and optional ReID feature
/// matching. ReID embeddings are supplied by the caller in detection order.
public final class BotSort: @unchecked Sendable {
    private let handle: UnsafeMutableRawPointer

    /// Creates a new BoT-SORT tracker.
    ///
    /// Negative values for `frameRate` and `trackBuffer` are clipped to `0`.
    ///
    /// - Parameters:
    ///   - frameRate: The input video frame rate. Defaults to `30`.
    ///   - trackBuffer: The number of frames to keep lost tracks. Defaults to `30`.
    ///   - trackHighThresh: The high-confidence detection threshold. Defaults to `0.6`.
    ///   - trackLowThresh: The lowest detection threshold used for association. Defaults to `0.1`.
    ///   - newTrackThresh: The threshold for starting new tracks. Defaults to `0.7`.
    ///   - matchThresh: The first association threshold. Defaults to `0.8`.
    ///   - useReId: Whether to use externally supplied ReID features. Defaults to `false`.
    ///   - proximityThresh: The IoU-distance gate for ReID matching. Defaults to `0.5`.
    ///   - appearanceThresh: The appearance-distance gate for ReID matching. Defaults to `0.25`.
    ///   - mot20: Whether to use MOT20 matching behavior. Defaults to `false`.
    ///   - useEcc: Whether to enable Rust ECC camera compensation. Defaults to `false`.
    public init(
        frameRate: Int = 30,
        trackBuffer: Int = 30,
        trackHighThresh: Float = 0.6,
        trackLowThresh: Float = 0.1,
        newTrackThresh: Float = 0.7,
        matchThresh: Float = 0.8,
        useReId: Bool = false,
        proximityThresh: Float = 0.5,
        appearanceThresh: Float = 0.25,
        mot20: Bool = false,
        useEcc: Bool = false
    ) {
        let h = jamtrack_bot_sort_create_with_config(
            max(frameRate, 0),
            max(trackBuffer, 0),
            trackHighThresh,
            trackLowThresh,
            newTrackThresh,
            matchThresh,
            useReId,
            proximityThresh,
            appearanceThresh,
            mot20,
            useEcc
        )
        precondition(h != nil, "Failed to create BotSort")
        self.handle = h!
    }

    deinit {
        jamtrack_bot_sort_drop(handle)
    }

    /// Updates the tracker with detections and returns tracked objects.
    ///
    /// Use this overload when ReID is disabled.
    public func update(_ objects: [TrackedObject]) -> Result<[TrackedObject], JamTrackError> {
        let cObjects = toCObjects(objects)
        var out = CObjectArray(data: nil, length: 0, _priv: nil)

        let status = cObjects.withUnsafeBufferPointer { buf in
            jamtrack_bot_sort_update(handle, buf.baseAddress, buf.count, &out)
        }
        defer { jamtrack_object_array_drop(&out) }

        if let error = statusToError(status) {
            return .failure(error)
        }
        return .success(fromCObjectArray(out))
    }

    /// Updates the tracker with detections and per-detection ReID embeddings.
    ///
    /// `features` must have the same count as `objects`; all feature rows must
    /// be non-empty and have the same dimension.
    public func update(
        _ objects: [TrackedObject],
        features: [[Float]]
    ) -> Result<[TrackedObject], JamTrackError> {
        guard features.count == objects.count else {
            return .failure(.invalidArgument)
        }

        let featureDim: Int
        if let first = features.first {
            guard !first.isEmpty else { return .failure(.invalidArgument) }
            featureDim = first.count
            guard features.allSatisfy({ $0.count == featureDim }) else {
                return .failure(.invalidArgument)
            }
        } else {
            featureDim = 0
        }

        let cObjects = toCObjects(objects)
        let flatFeatures = features.flatMap { $0 }
        var out = CObjectArray(data: nil, length: 0, _priv: nil)

        let status = cObjects.withUnsafeBufferPointer { objectBuf in
            flatFeatures.withUnsafeBufferPointer { featureBuf in
                jamtrack_bot_sort_update_with_features(
                    handle,
                    objectBuf.baseAddress,
                    objectBuf.count,
                    featureBuf.baseAddress,
                    featureDim,
                    &out
                )
            }
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
        let status = jamtrack_bot_sort_frame_count(handle, &value)
        precondition(status == JAMTRACK_STATUS_OK, "Unexpected FFI error: \(status)")
        return value
    }

    /// The number of active and lost tracks currently retained.
    public var trackerCount: Int {
        var value: Int = 0
        let status = jamtrack_bot_sort_tracker_count(handle, &value)
        precondition(status == JAMTRACK_STATUS_OK, "Unexpected FFI error: \(status)")
        return value
    }
}

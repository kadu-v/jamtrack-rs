import JamTrack

/// A multi-object tracker based on the ByteTrack algorithm.
///
/// ByteTrack associates detections across frames using a two-stage matching
/// strategy that leverages both high-confidence and low-confidence detections.
///
/// ```swift
/// let tracker = ByteTracker()
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
public final class ByteTracker: @unchecked Sendable {
    private let handle: UnsafeMutableRawPointer

    /// Creates a new ByteTracker.
    ///
    /// - Parameters:
    ///   - frameRate: The frame rate of the video. Defaults to `30`.
    ///   - trackBuffer: The number of frames to buffer lost tracks. Defaults to `30`.
    ///   - trackThresh: The detection threshold for creating new tracks. Defaults to `0.5`.
    ///   - highThresh: The threshold for high-confidence detections. Defaults to `0.6`.
    ///   - matchThresh: The IoU threshold for matching. Defaults to `0.8`.
    public init(
        frameRate: Int = 30,
        trackBuffer: Int = 30,
        trackThresh: Float = 0.5,
        highThresh: Float = 0.6,
        matchThresh: Float = 0.8
    ) {
        let h = jamtrack_byte_tracker_create(
            frameRate, trackBuffer, trackThresh, highThresh, matchThresh
        )
        precondition(h != nil, "Failed to create ByteTracker")
        self.handle = h!
    }

    deinit {
        jamtrack_byte_tracker_drop(handle)
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
            jamtrack_byte_tracker_update(handle, buf.baseAddress, buf.count, &out)
        }
        defer { jamtrack_object_array_drop(&out) }

        if let error = statusToError(status) {
            return .failure(error)
        }
        return .success(fromCObjectArray(out))
    }
}

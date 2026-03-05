import JamTrack

/// A multi-object tracker based on the OC-SORT algorithm.
///
/// OC-SORT improves upon SORT by using observation-centric online smoothing
/// and incorporating velocity direction consistency for more robust association
/// under occlusion.
///
/// ```swift
/// let tracker = OCSort()
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
public final class OCSort: @unchecked Sendable {
    private let handle: UnsafeMutableRawPointer

    /// Creates a new OC-SORT tracker.
    ///
    /// Negative values for `maxAge`, `minHits`, and `deltaT` are clipped to `0`.
    ///
    /// - Parameters:
    ///   - detThresh: The detection confidence threshold. Defaults to `0.5`.
    ///   - maxAge: The maximum number of frames a track can be lost before removal. Defaults to `30`.
    ///   - minHits: The minimum number of hits before a track is reported. Defaults to `3`.
    ///   - iouThreshold: The IoU threshold for matching. Defaults to `0.3`.
    ///   - deltaT: The time step for velocity estimation. Defaults to `3`.
    ///   - inertia: The inertia weight for velocity direction consistency. Defaults to `0.2`.
    ///   - useByte: Whether to use the BYTE association strategy for low-confidence detections. Defaults to `false`.
    public init(
        detThresh: Float = 0.5,
        maxAge: Int = 30,
        minHits: Int = 3,
        iouThreshold: Float = 0.3,
        deltaT: Int = 3,
        inertia: Float = 0.2,
        useByte: Bool = false
    ) {
        let h = jamtrack_oc_sort_create_with_config(
            detThresh,
            max(maxAge, 0),
            max(minHits, 0),
            iouThreshold,
            max(deltaT, 0),
            inertia,
            useByte
        )
        precondition(h != nil, "Failed to create OCSort")
        self.handle = h!
    }

    deinit {
        jamtrack_oc_sort_drop(handle)
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
            jamtrack_oc_sort_update(handle, buf.baseAddress, buf.count, &out)
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
        let status = jamtrack_oc_sort_frame_count(handle, &value)
        precondition(status == JAMTRACK_STATUS_OK, "Unexpected FFI error: \(status)")
        return value
    }

    /// The number of active tracks.
    public var trackerCount: Int {
        var value: Int = 0
        let status = jamtrack_oc_sort_tracker_count(handle, &value)
        precondition(status == JAMTRACK_STATUS_OK, "Unexpected FFI error: \(status)")
        return value
    }
}

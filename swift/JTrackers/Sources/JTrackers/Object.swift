import JamTrack

// MARK: - TrackedObject

/// An object detected and tracked across frames.
///
/// Each ``TrackedObject`` holds a bounding box (`x`, `y`, `width`, `height`),
/// a detection confidence (`prob`), and an optional track identifier (`trackId`)
/// assigned by the tracker.
///
/// ```swift
/// // As input to update (trackId is ignored)
/// let det = TrackedObject(x: 10, y: 20, width: 100, height: 200, prob: 0.9)
///
/// // Output from update (trackId is assigned by the tracker)
/// let results = tracker.update([det])
/// for obj in try results.get() {
///     print("Track \(obj.trackId ?? -1): \(obj.x), \(obj.y)")
/// }
/// ```
public struct TrackedObject: Sendable, Equatable {
    /// The x-coordinate of the bounding box top-left corner.
    public let x: Float
    /// The y-coordinate of the bounding box top-left corner.
    public let y: Float
    /// The width of the bounding box.
    public let width: Float
    /// The height of the bounding box.
    public let height: Float
    /// The detection confidence score in `[0, 1]`.
    public let prob: Float
    /// The track identifier assigned by the tracker, or `nil` when used as input.
    public let trackId: Int?

    /// Creates a new tracked object.
    ///
    /// - Parameters:
    ///   - x: The x-coordinate of the bounding box top-left corner.
    ///   - y: The y-coordinate of the bounding box top-left corner.
    ///   - width: The width of the bounding box.
    ///   - height: The height of the bounding box.
    ///   - prob: The detection confidence score.
    ///   - trackId: The track identifier. Defaults to `nil`.
    public init(x: Float, y: Float, width: Float, height: Float, prob: Float, trackId: Int? = nil) {
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.prob = prob
        self.trackId = trackId
    }
}

// MARK: - JamTrackError

/// Errors returned by tracker operations.
public enum JamTrackError: Error, Sendable, Equatable {
    /// A null pointer was encountered in the FFI layer.
    case nullPointer
    /// An invalid argument was passed to the FFI layer.
    case invalidArgument
    /// An internal error occurred in the Rust tracking engine.
    case internalError
    /// An unknown FFI status code was returned.
    case unknownError(Int32)
    /// The tracker handle could not be created.
    case createFailed
}

// MARK: - Internal helpers

func statusToError(_ code: Int32) -> JamTrackError? {
    switch code {
    case JAMTRACK_STATUS_OK:
        return nil
    case JAMTRACK_STATUS_NULL_POINTER:
        return .nullPointer
    case JAMTRACK_STATUS_INVALID_ARG:
        return .invalidArgument
    case JAMTRACK_STATUS_INTERNAL_ERROR:
        return .internalError
    default:
        return .unknownError(code)
    }
}

func toCObjects(_ objects: [TrackedObject]) -> [CObject] {
    objects.map { obj in
        CObject(
            x: obj.x,
            y: obj.y,
            width: obj.width,
            height: obj.height,
            prob: obj.prob,
            track_id: obj.trackId.map { Int32($0) } ?? -1
        )
    }
}

func fromCObjectArray(_ array: CObjectArray) -> [TrackedObject] {
    guard array.length > 0, array.data != nil else { return [] }
    let buffer = UnsafeBufferPointer(start: array.data, count: array.length)
    return buffer.map { c in
        TrackedObject(
            x: c.x,
            y: c.y,
            width: c.width,
            height: c.height,
            prob: c.prob,
            trackId: c.track_id >= 0 ? Int(c.track_id) : nil
        )
    }
}

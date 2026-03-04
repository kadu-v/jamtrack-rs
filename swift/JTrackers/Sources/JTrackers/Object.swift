import JamTrack

// MARK: - TrackedObject

public struct TrackedObject: Sendable, Equatable {
    public let x: Float
    public let y: Float
    public let width: Float
    public let height: Float
    public let prob: Float
    public let trackId: Int?

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

public enum JamTrackError: Error, Sendable, Equatable {
    case nullPointer
    case invalidArgument
    case internalError
    case unknownError(Int32)
    case createFailed
}

// MARK: - Internal helpers

func checkStatus(_ code: Int32) throws {
    switch code {
    case JAMTRACK_STATUS_OK:
        return
    case JAMTRACK_STATUS_NULL_POINTER:
        throw JamTrackError.nullPointer
    case JAMTRACK_STATUS_INVALID_ARG:
        throw JamTrackError.invalidArgument
    case JAMTRACK_STATUS_INTERNAL_ERROR:
        throw JamTrackError.internalError
    default:
        throw JamTrackError.unknownError(code)
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

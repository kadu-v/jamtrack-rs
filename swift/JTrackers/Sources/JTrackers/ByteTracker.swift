import JamTrack

public final class ByteTracker: @unchecked Sendable {
    private var handle: UnsafeMutableRawPointer?

    public init(
        frameRate: Int = 30,
        trackBuffer: Int = 30,
        trackThresh: Float = 0.5,
        highThresh: Float = 0.6,
        matchThresh: Float = 0.8
    ) throws {
        let h = jamtrack_byte_tracker_create(
            frameRate, trackBuffer, trackThresh, highThresh, matchThresh
        )
        guard let h else { throw JamTrackError.createFailed }
        self.handle = h
    }

    deinit {
        if let h = handle {
            jamtrack_byte_tracker_drop(h)
        }
    }

    public func update(_ objects: [TrackedObject]) throws -> [TrackedObject] {
        let cObjects = toCObjects(objects)
        var out = CObjectArray(data: nil, length: 0, _priv: nil)

        let status = cObjects.withUnsafeBufferPointer { buf in
            jamtrack_byte_tracker_update(handle, buf.baseAddress, buf.count, &out)
        }
        defer { jamtrack_object_array_drop(&out) }

        try checkStatus(status)
        return fromCObjectArray(out)
    }
}

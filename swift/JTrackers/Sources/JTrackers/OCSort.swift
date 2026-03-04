import JamTrack

public final class OCSort: @unchecked Sendable {
    private var handle: UnsafeMutableRawPointer?

    public init(
        detThresh: Float = 0.5,
        maxAge: Int = 30,
        minHits: Int = 3,
        iouThreshold: Float = 0.3,
        deltaT: Int = 3,
        inertia: Float = 0.2,
        useByte: Bool = false
    ) throws {
        guard maxAge >= 0, minHits >= 0, deltaT >= 0 else {
            throw JamTrackError.invalidArgument
        }
        let h = jamtrack_oc_sort_create_with_config(
            detThresh,
            maxAge,
            minHits,
            iouThreshold,
            deltaT,
            inertia,
            useByte
        )
        guard let h else { throw JamTrackError.createFailed }
        self.handle = h
    }

    deinit {
        if let h = handle {
            jamtrack_oc_sort_drop(h)
        }
    }

    public func update(_ objects: [TrackedObject]) throws -> [TrackedObject] {
        let cObjects = toCObjects(objects)
        var out = CObjectArray(data: nil, length: 0, _priv: nil)

        let status = cObjects.withUnsafeBufferPointer { buf in
            jamtrack_oc_sort_update(handle, buf.baseAddress, buf.count, &out)
        }
        defer { jamtrack_object_array_drop(&out) }

        try checkStatus(status)
        return fromCObjectArray(out)
    }

    public var frameCount: Int {
        get throws {
            var value: Int = 0
            let status = jamtrack_oc_sort_frame_count(handle, &value)
            try checkStatus(status)
            return value
        }
    }

    public var trackerCount: Int {
        get throws {
            var value: Int = 0
            let status = jamtrack_oc_sort_tracker_count(handle, &value)
            try checkStatus(status)
            return value
        }
    }
}

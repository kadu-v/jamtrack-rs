import JamTrack

public final class BoostTracker: @unchecked Sendable {
    private var handle: UnsafeMutableRawPointer?

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
    ) throws {
        guard maxAge >= 0, minHits >= 0 else {
            throw JamTrackError.invalidArgument
        }
        let h = jamtrack_boost_tracker_create_with_config(
            detThresh,
            iouThreshold,
            maxAge,
            minHits,
            lambdaIou,
            lambdaMhd,
            lambdaShape,
            useDloBoost,
            useDuoBoost,
            enableBoostPlus,
            enableBoostPlusPlus,
            useShapeSimilarityV1
        )
        guard let h else { throw JamTrackError.createFailed }
        self.handle = h
    }

    deinit {
        if let h = handle {
            jamtrack_boost_tracker_drop(h)
        }
    }

    public func update(_ objects: [TrackedObject]) throws -> [TrackedObject] {
        let cObjects = toCObjects(objects)
        var out = CObjectArray(data: nil, length: 0, _priv: nil)

        let status = cObjects.withUnsafeBufferPointer { buf in
            jamtrack_boost_tracker_update(handle, buf.baseAddress, buf.count, &out)
        }
        defer { jamtrack_object_array_drop(&out) }

        try checkStatus(status)
        return fromCObjectArray(out)
    }

    public var frameCount: Int {
        get throws {
            var value: Int = 0
            let status = jamtrack_boost_tracker_frame_count(handle, &value)
            try checkStatus(status)
            return value
        }
    }

    public var trackerCount: Int {
        get throws {
            var value: Int = 0
            let status = jamtrack_boost_tracker_tracker_count(handle, &value)
            try checkStatus(status)
            return value
        }
    }
}

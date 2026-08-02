import JamTrack

/// A point used to define a FastTracker road-region constraint.
public struct FastTrackerPoint: Sendable, Equatable {
    public let x: Float
    public let y: Float

    public init(x: Float, y: Float) {
        self.x = x
        self.y = y
    }
}

/// A four-point road region ordered as `(E1, E2, O2, O1)`.
public struct FastTrackerRoi: Sendable, Equatable {
    public let e1: FastTrackerPoint
    public let e2: FastTrackerPoint
    public let o2: FastTrackerPoint
    public let o1: FastTrackerPoint

    public init(
        e1: FastTrackerPoint,
        e2: FastTrackerPoint,
        o2: FastTrackerPoint,
        o1: FastTrackerPoint
    ) {
        self.e1 = e1
        self.e2 = e2
        self.o2 = o2
        self.o1 = o1
    }

    fileprivate var cValue: CFastTrackerRoi {
        CFastTrackerRoi(
            e1: CFastTrackerPoint(x: e1.x, y: e1.y),
            e2: CFastTrackerPoint(x: e2.x, y: e2.y),
            o2: CFastTrackerPoint(x: o2.x, y: o2.y),
            o1: CFastTrackerPoint(x: o1.x, y: o1.y)
        )
    }
}

/// FastTracker preserves identities through heavy occlusion and can constrain
/// trajectories using four-point road regions.
public final class FastTracker: @unchecked Sendable {
    private let handle: UnsafeMutableRawPointer

    public init(
        frameRate: Int = 30,
        trackBuffer: Int = 30,
        trackThresh: Float = 0.6,
        matchThresh: Float = 0.7,
        resetVelocityOffset: Int = 5,
        resetPositionOffset: Int = 3,
        enlargeBbox: Float = 1.2,
        dampenMotion: Float = 0.85,
        activeOcclusionToLost: Int = 15,
        initIouSuppression: Float = 0.8,
        rois: [FastTrackerRoi] = [],
        roiRepairMaxGap: Int = 15,
        directionWindow: Int = 10,
        directionMarginDegrees: Float = 2.0,
        mot20: Bool = false
    ) {
        let cRois = rois.map(\.cValue)
        let created = cRois.withUnsafeBufferPointer { buffer in
            jamtrack_fast_tracker_create_with_config(
                max(frameRate, 0),
                max(trackBuffer, 0),
                trackThresh,
                matchThresh,
                max(resetVelocityOffset, 0),
                max(resetPositionOffset, 0),
                enlargeBbox,
                dampenMotion,
                max(activeOcclusionToLost, 0),
                initIouSuppression,
                buffer.baseAddress,
                buffer.count,
                max(roiRepairMaxGap, 0),
                max(directionWindow, 0),
                directionMarginDegrees,
                mot20
            )
        }
        precondition(created != nil, "Failed to create FastTracker")
        handle = created!
    }

    deinit {
        jamtrack_fast_tracker_drop(handle)
    }

    public func update(_ objects: [TrackedObject]) -> Result<[TrackedObject], JamTrackError> {
        let cObjects = toCObjects(objects)
        var out = CObjectArray(data: nil, length: 0, _priv: nil)
        let status = cObjects.withUnsafeBufferPointer { buffer in
            jamtrack_fast_tracker_update(
                handle,
                buffer.baseAddress,
                buffer.count,
                &out
            )
        }
        defer { jamtrack_object_array_drop(&out) }
        if let error = statusToError(status) {
            return .failure(error)
        }
        return .success(fromCObjectArray(out))
    }

    public var frameCount: Int {
        var value = 0
        let status = jamtrack_fast_tracker_frame_count(handle, &value)
        precondition(status == JAMTRACK_STATUS_OK, "Unexpected FFI error: \(status)")
        return value
    }

    public var trackerCount: Int {
        var value = 0
        let status = jamtrack_fast_tracker_tracker_count(handle, &value)
        precondition(status == JAMTRACK_STATUS_OK, "Unexpected FFI error: \(status)")
        return value
    }
}

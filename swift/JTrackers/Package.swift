// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "JTrackers",
    platforms: [.macOS(.v12), .iOS(.v15)],
    products: [
        .library(name: "JTrackers", targets: ["JTrackers"]),
    ],
    targets: [
        .binaryTarget(
            name: "JamTrack",
            path: "Artifacts/JamTrack.xcframework"
        ),
        .target(
            name: "JTrackers",
            dependencies: ["JamTrack"]
        ),
        .testTarget(
            name: "JTrackersTests",
            dependencies: ["JTrackers"]
        ),
    ]
)

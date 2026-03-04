// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "JTrackers",
    platforms: [.macOS(.v12), .iOS(.v15)],
    products: [
        .library(name: "JTrackers", targets: ["JTrackers"]),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-docc-plugin", from: "1.0.0"),
    ],
    targets: [
        .binaryTarget(
            name: "JamTrack",
            path: "Artifacts/JamTrack.xcframework.zip"
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

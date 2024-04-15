use crate::rect::Rect;
use crate::strack::STrack;

#[test]
fn test_joint_strack() {
    use crate::byte_tracker::{self, ByteTracker};

    let dummy_rect = Rect::new(0.0, 0.0, 0.0, 0.0);
    let a_tlist: Vec<STrack> = vec![
        STrack::dummy_strack(1),
        STrack::dummy_strack(2),
        STrack::dummy_strack(3),
        STrack::dummy_strack(4),
        STrack::dummy_strack(5),
    ];
    let b_tlist: Vec<STrack> = vec![
        STrack::dummy_strack(2),
        STrack::dummy_strack(3),
        STrack::dummy_strack(4),
        STrack::dummy_strack(5),
        STrack::dummy_strack(6),
    ];

    let byte_tracker = ByteTracker::new(0.0, 0.0, 0.0, 0);
    let result = byte_tracker.joint_strack(&a_tlist, &b_tlist);

    let expected_result: Vec<STrack> = vec![
        STrack::dummy_strack(1),
        STrack::dummy_strack(2),
        STrack::dummy_strack(3),
        STrack::dummy_strack(4),
        STrack::dummy_strack(5),
        STrack::dummy_strack(6),
    ];

    assert_eq!(result, expected_result);
}

use crate::byte_tracker::ByteTracker;
use crate::strack::STrack;

#[test]
fn test_joint_strack() {
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

    let result = ByteTracker::joint_stracks(&a_tlist, &b_tlist);

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

#[test]
pub fn test_sub_stracks() {
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

    let result = ByteTracker::sub_stracks(&a_tlist, &b_tlist);
    let expected_result: Vec<STrack> = vec![STrack::dummy_strack(1)];

    assert_eq!(result, expected_result);
}

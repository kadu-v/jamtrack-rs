use std::panic::catch_unwind;

use crate::object::Object;
use crate::rect::Rect;
use crate::{BoostTracker, ByteTracker, OCSort};

// ---------------------------------------------------------------------------
// Status codes
// ---------------------------------------------------------------------------

const STATUS_OK: i32 = 0;
const STATUS_NULL_POINTER: i32 = 1;
#[allow(dead_code)]
const STATUS_INVALID_ARGUMENT: i32 = 2;
const STATUS_INTERNAL_ERROR: i32 = 3;

// ---------------------------------------------------------------------------
// C-compatible data types
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CObject {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub prob: f32,
    pub track_id: i32,
}

#[repr(C)]
pub struct CObjectArray {
    pub data: *const CObject,
    pub length: usize,
    pub _priv: *mut core::ffi::c_void,
}

// ---------------------------------------------------------------------------
// Conversion helpers
// ---------------------------------------------------------------------------

fn cobject_to_object(src: &CObject) -> Object {
    let rect = Rect::new(src.x, src.y, src.width, src.height);
    let track_id = if src.track_id < 0 {
        None
    } else {
        Some(src.track_id as usize)
    };
    Object::new(rect, src.prob, track_id)
}

fn object_to_cobject(src: &Object) -> CObject {
    CObject {
        x: src.get_x(),
        y: src.get_y(),
        width: src.get_width(),
        height: src.get_height(),
        prob: src.get_prob(),
        track_id: match src.get_track_id() {
            Some(id) => id as i32,
            None => -1,
        },
    }
}

fn slice_from_raw<'a>(ptr: *const CObject, len: usize) -> Result<&'a [CObject], i32> {
    if len == 0 {
        return Ok(&[]);
    }
    if ptr.is_null() {
        return Err(STATUS_NULL_POINTER);
    }
    Ok(unsafe { std::slice::from_raw_parts(ptr, len) })
}

fn write_object_array(out: *mut CObjectArray, items: Vec<CObject>) -> i32 {
    if out.is_null() {
        return STATUS_NULL_POINTER;
    }
    let len = items.len();
    let boxed: Box<[CObject]> = items.into_boxed_slice();
    let data_ptr = boxed.as_ptr();
    let priv_ptr = Box::into_raw(boxed) as *mut core::ffi::c_void;
    unsafe {
        (*out).data = data_ptr;
        (*out).length = len;
        (*out)._priv = priv_ptr;
    }
    STATUS_OK
}

fn free_object_array_inner(array: &mut CObjectArray) {
    if !array._priv.is_null() {
        let raw = array._priv as *mut CObject;
        let len = array.length;
        unsafe {
            let _ = Box::from_raw(std::slice::from_raw_parts_mut(raw, len));
        }
    }
    array.data = std::ptr::null();
    array.length = 0;
    array._priv = std::ptr::null_mut();
}

/// Generate a read-only accessor FFI function that calls `$method` on `$TrackerType`.
macro_rules! ffi_accessor {
    ($fn_name:ident, $TrackerType:ty, $method:ident) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $fn_name(
            handle: *mut core::ffi::c_void,
            out_value: *mut usize,
        ) -> i32 {
            if handle.is_null() || out_value.is_null() {
                return STATUS_NULL_POINTER;
            }
            let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
                let tracker = unsafe { &*(handle as *const $TrackerType) };
                unsafe { *out_value = tracker.$method() };
                STATUS_OK
            }));
            match result {
                Ok(code) => code,
                Err(_) => STATUS_INTERNAL_ERROR,
            }
        }
    };
}

// ---------------------------------------------------------------------------
// ByteTracker FFI
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "C" fn jamtrack_byte_tracker_create(
    frame_rate: usize,
    track_buffer: usize,
    track_thresh: f32,
    high_thresh: f32,
    match_thresh: f32,
) -> *mut core::ffi::c_void {
    let result = catch_unwind(|| {
        let tracker = ByteTracker::new(
            frame_rate,
            track_buffer,
            track_thresh,
            high_thresh,
            match_thresh,
        );
        Box::into_raw(Box::new(tracker)) as *mut core::ffi::c_void
    });
    match result {
        Ok(ptr) => ptr,
        Err(_) => std::ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn jamtrack_byte_tracker_update(
    handle: *mut core::ffi::c_void,
    objects: *const CObject,
    length: usize,
    out_array: *mut CObjectArray,
) -> i32 {
    if handle.is_null() {
        return STATUS_NULL_POINTER;
    }
    if out_array.is_null() {
        return STATUS_NULL_POINTER;
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let tracker = unsafe { &mut *(handle as *mut ByteTracker) };
        let c_slice = match slice_from_raw(objects, length) {
            Ok(s) => s,
            Err(code) => return code,
        };
        let input: Vec<Object> = c_slice.iter().map(cobject_to_object).collect();
        match tracker.update(&input) {
            Ok(results) => {
                let c_results: Vec<CObject> = results.iter().map(object_to_cobject).collect();
                write_object_array(out_array, c_results)
            }
            Err(_) => STATUS_INTERNAL_ERROR,
        }
    }));

    match result {
        Ok(code) => code,
        Err(_) => STATUS_INTERNAL_ERROR,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn jamtrack_byte_tracker_drop(handle: *mut core::ffi::c_void) {
    if handle.is_null() {
        return;
    }
    let _ = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = unsafe { Box::from_raw(handle as *mut ByteTracker) };
    }));
}

// ---------------------------------------------------------------------------
// OCSort FFI
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "C" fn jamtrack_oc_sort_create(
    det_thresh: f32,
) -> *mut core::ffi::c_void {
    let result = catch_unwind(|| {
        let tracker = OCSort::new(det_thresh);
        Box::into_raw(Box::new(tracker)) as *mut core::ffi::c_void
    });
    match result {
        Ok(ptr) => ptr,
        Err(_) => std::ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn jamtrack_oc_sort_create_with_config(
    det_thresh: f32,
    max_age: usize,
    min_hits: usize,
    iou_threshold: f32,
    delta_t: usize,
    inertia: f32,
    use_byte: bool,
) -> *mut core::ffi::c_void {
    let result = catch_unwind(|| {
        let tracker = OCSort::new(det_thresh)
            .with_max_age(max_age)
            .with_min_hits(min_hits)
            .with_iou_threshold(iou_threshold)
            .with_delta_t(delta_t)
            .with_inertia(inertia)
            .with_byte(use_byte);
        Box::into_raw(Box::new(tracker)) as *mut core::ffi::c_void
    });
    match result {
        Ok(ptr) => ptr,
        Err(_) => std::ptr::null_mut(),
    }
}

ffi_accessor!(jamtrack_oc_sort_frame_count, OCSort, frame_count);
ffi_accessor!(jamtrack_oc_sort_tracker_count, OCSort, tracker_count);

#[unsafe(no_mangle)]
pub unsafe extern "C" fn jamtrack_oc_sort_update(
    handle: *mut core::ffi::c_void,
    objects: *const CObject,
    length: usize,
    out_array: *mut CObjectArray,
) -> i32 {
    if handle.is_null() {
        return STATUS_NULL_POINTER;
    }
    if out_array.is_null() {
        return STATUS_NULL_POINTER;
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let tracker = unsafe { &mut *(handle as *mut OCSort) };
        let c_slice = match slice_from_raw(objects, length) {
            Ok(s) => s,
            Err(code) => return code,
        };
        let input: Vec<Object> = c_slice.iter().map(cobject_to_object).collect();
        match tracker.update(&input) {
            Ok(results) => {
                let c_results: Vec<CObject> = results.iter().map(object_to_cobject).collect();
                write_object_array(out_array, c_results)
            }
            Err(_) => STATUS_INTERNAL_ERROR,
        }
    }));

    match result {
        Ok(code) => code,
        Err(_) => STATUS_INTERNAL_ERROR,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn jamtrack_oc_sort_drop(handle: *mut core::ffi::c_void) {
    if handle.is_null() {
        return;
    }
    let _ = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = unsafe { Box::from_raw(handle as *mut OCSort) };
    }));
}

// ---------------------------------------------------------------------------
// BoostTracker FFI
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "C" fn jamtrack_boost_tracker_create(
    det_thresh: f32,
    iou_threshold: f32,
    max_age: usize,
    min_hits: usize,
) -> *mut core::ffi::c_void {
    let result = catch_unwind(|| {
        let tracker = BoostTracker::new(det_thresh, iou_threshold, max_age, min_hits);
        Box::into_raw(Box::new(tracker)) as *mut core::ffi::c_void
    });
    match result {
        Ok(ptr) => ptr,
        Err(_) => std::ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn jamtrack_boost_tracker_create_with_config(
    det_thresh: f32,
    iou_threshold: f32,
    max_age: usize,
    min_hits: usize,
    lambda_iou: f32,
    lambda_mhd: f32,
    lambda_shape: f32,
    use_dlo_boost: bool,
    use_duo_boost: bool,
    enable_boost_plus: bool,
    enable_boost_plus_plus: bool,
    use_shape_similarity_v1: bool,
) -> *mut core::ffi::c_void {
    let result = catch_unwind(|| {
        let mut tracker = BoostTracker::new(det_thresh, iou_threshold, max_age, min_hits)
            .with_lambdas(lambda_iou, lambda_mhd, lambda_shape)
            .with_boost(use_dlo_boost, use_duo_boost);

        if enable_boost_plus_plus {
            tracker = tracker.with_boost_plus_plus();
        } else if enable_boost_plus {
            tracker = tracker.with_boost_plus();
        }

        if use_shape_similarity_v1 {
            tracker = tracker.with_shape_similarity_v1();
        }

        Box::into_raw(Box::new(tracker)) as *mut core::ffi::c_void
    });
    match result {
        Ok(ptr) => ptr,
        Err(_) => std::ptr::null_mut(),
    }
}

ffi_accessor!(jamtrack_boost_tracker_frame_count, BoostTracker, frame_count);
ffi_accessor!(jamtrack_boost_tracker_tracker_count, BoostTracker, tracker_count);

#[unsafe(no_mangle)]
pub unsafe extern "C" fn jamtrack_boost_tracker_update(
    handle: *mut core::ffi::c_void,
    objects: *const CObject,
    length: usize,
    out_array: *mut CObjectArray,
) -> i32 {
    if handle.is_null() {
        return STATUS_NULL_POINTER;
    }
    if out_array.is_null() {
        return STATUS_NULL_POINTER;
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let tracker = unsafe { &mut *(handle as *mut BoostTracker) };
        let c_slice = match slice_from_raw(objects, length) {
            Ok(s) => s,
            Err(code) => return code,
        };
        let input: Vec<Object> = c_slice.iter().map(cobject_to_object).collect();
        match tracker.update(&input) {
            Ok(results) => {
                let c_results: Vec<CObject> = results.iter().map(object_to_cobject).collect();
                write_object_array(out_array, c_results)
            }
            Err(_) => STATUS_INTERNAL_ERROR,
        }
    }));

    match result {
        Ok(code) => code,
        Err(_) => STATUS_INTERNAL_ERROR,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn jamtrack_boost_tracker_drop(handle: *mut core::ffi::c_void) {
    if handle.is_null() {
        return;
    }
    let _ = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = unsafe { Box::from_raw(handle as *mut BoostTracker) };
    }));
}

// ---------------------------------------------------------------------------
// Shared: CObjectArray drop
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "C" fn jamtrack_object_array_drop(array: *mut CObjectArray) {
    if array.is_null() {
        return;
    }
    let _ = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let arr = unsafe { &mut *array };
        free_object_array_inner(arr);
    }));
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_cobject(x: f32, y: f32, w: f32, h: f32, prob: f32) -> CObject {
        CObject {
            x,
            y,
            width: w,
            height: h,
            prob,
            track_id: -1,
        }
    }

    #[test]
    fn test_cobject_to_object_roundtrip() {
        let c = CObject {
            x: 10.0,
            y: 20.0,
            width: 100.0,
            height: 200.0,
            prob: 0.9,
            track_id: 5,
        };
        let obj = cobject_to_object(&c);
        assert_eq!(obj.get_x(), 10.0);
        assert_eq!(obj.get_y(), 20.0);
        assert_eq!(obj.get_width(), 100.0);
        assert_eq!(obj.get_height(), 200.0);
        assert_eq!(obj.get_prob(), 0.9);
        assert_eq!(obj.get_track_id(), Some(5));

        let back = object_to_cobject(&obj);
        assert_eq!(back.x, 10.0);
        assert_eq!(back.track_id, 5);
    }

    #[test]
    fn test_cobject_none_track_id() {
        let c = CObject {
            x: 0.0,
            y: 0.0,
            width: 50.0,
            height: 50.0,
            prob: 0.5,
            track_id: -1,
        };
        let obj = cobject_to_object(&c);
        assert_eq!(obj.get_track_id(), None);

        let back = object_to_cobject(&obj);
        assert_eq!(back.track_id, -1);
    }

    #[test]
    fn test_byte_tracker_create_update_drop() {
        unsafe {
            let handle = jamtrack_byte_tracker_create(30, 30, 0.5, 0.6, 0.8);
            assert!(!handle.is_null());

            let objects = [
                make_test_cobject(10.0, 20.0, 100.0, 200.0, 0.9),
                make_test_cobject(50.0, 60.0, 80.0, 160.0, 0.8),
            ];

            let mut out = CObjectArray {
                data: std::ptr::null(),
                length: 0,
                _priv: std::ptr::null_mut(),
            };

            let status =
                jamtrack_byte_tracker_update(handle, objects.as_ptr(), objects.len(), &mut out);
            assert_eq!(status, STATUS_OK);
            assert!(out.length > 0);

            jamtrack_object_array_drop(&mut out);
            assert!(out.data.is_null());
            assert_eq!(out.length, 0);

            jamtrack_byte_tracker_drop(handle);
        }
    }

    #[test]
    fn test_oc_sort_create_update_drop() {
        unsafe {
            let handle = jamtrack_oc_sort_create(0.5);
            assert!(!handle.is_null());

            let objects = [
                make_test_cobject(10.0, 20.0, 100.0, 200.0, 0.9),
            ];

            let mut out = CObjectArray {
                data: std::ptr::null(),
                length: 0,
                _priv: std::ptr::null_mut(),
            };

            let status =
                jamtrack_oc_sort_update(handle, objects.as_ptr(), objects.len(), &mut out);
            assert_eq!(status, STATUS_OK);

            jamtrack_object_array_drop(&mut out);
            jamtrack_oc_sort_drop(handle);
        }
    }

    #[test]
    fn test_boost_tracker_create_update_drop() {
        unsafe {
            let handle = jamtrack_boost_tracker_create(0.5, 0.3, 30, 3);
            assert!(!handle.is_null());

            let objects = [
                make_test_cobject(10.0, 20.0, 100.0, 200.0, 0.9),
            ];

            let mut out = CObjectArray {
                data: std::ptr::null(),
                length: 0,
                _priv: std::ptr::null_mut(),
            };

            let status = jamtrack_boost_tracker_update(
                handle,
                objects.as_ptr(),
                objects.len(),
                &mut out,
            );
            assert_eq!(status, STATUS_OK);

            jamtrack_object_array_drop(&mut out);
            jamtrack_boost_tracker_drop(handle);
        }
    }

    #[test]
    fn test_empty_input() {
        unsafe {
            let handle = jamtrack_byte_tracker_create(30, 30, 0.5, 0.6, 0.8);
            assert!(!handle.is_null());

            let mut out = CObjectArray {
                data: std::ptr::null(),
                length: 0,
                _priv: std::ptr::null_mut(),
            };

            // length == 0, objects == null should be fine
            let status =
                jamtrack_byte_tracker_update(handle, std::ptr::null(), 0, &mut out);
            assert_eq!(status, STATUS_OK);
            assert_eq!(out.length, 0);

            jamtrack_object_array_drop(&mut out);
            jamtrack_byte_tracker_drop(handle);
        }
    }

    #[test]
    fn test_null_handle() {
        unsafe {
            let mut out = CObjectArray {
                data: std::ptr::null(),
                length: 0,
                _priv: std::ptr::null_mut(),
            };

            let status = jamtrack_byte_tracker_update(
                std::ptr::null_mut(),
                std::ptr::null(),
                0,
                &mut out,
            );
            assert_eq!(status, STATUS_NULL_POINTER);

            let status = jamtrack_oc_sort_update(
                std::ptr::null_mut(),
                std::ptr::null(),
                0,
                &mut out,
            );
            assert_eq!(status, STATUS_NULL_POINTER);

            let status = jamtrack_boost_tracker_update(
                std::ptr::null_mut(),
                std::ptr::null(),
                0,
                &mut out,
            );
            assert_eq!(status, STATUS_NULL_POINTER);
        }
    }

    #[test]
    fn test_null_out_array() {
        unsafe {
            let handle = jamtrack_byte_tracker_create(30, 30, 0.5, 0.6, 0.8);
            assert!(!handle.is_null());

            let status = jamtrack_byte_tracker_update(
                handle,
                std::ptr::null(),
                0,
                std::ptr::null_mut(),
            );
            assert_eq!(status, STATUS_NULL_POINTER);

            jamtrack_byte_tracker_drop(handle);
        }
    }

    #[test]
    fn test_drop_null_handle_is_noop() {
        unsafe {
            jamtrack_byte_tracker_drop(std::ptr::null_mut());
            jamtrack_oc_sort_drop(std::ptr::null_mut());
            jamtrack_boost_tracker_drop(std::ptr::null_mut());
        }
    }

    #[test]
    fn test_drop_null_array_is_noop() {
        unsafe {
            jamtrack_object_array_drop(std::ptr::null_mut());
        }
    }

    // -----------------------------------------------------------------------
    // OCSort create_with_config + accessors
    // -----------------------------------------------------------------------

    #[test]
    fn test_oc_sort_create_with_config() {
        unsafe {
            let handle =
                jamtrack_oc_sort_create_with_config(0.6, 20, 2, 0.4, 5, 0.3, true);
            assert!(!handle.is_null());
            jamtrack_oc_sort_drop(handle);
        }
    }

    #[test]
    fn test_oc_sort_frame_count_initial() {
        unsafe {
            let handle = jamtrack_oc_sort_create(0.5);
            assert!(!handle.is_null());

            let mut value: usize = 999;
            let status = jamtrack_oc_sort_frame_count(handle, &mut value);
            assert_eq!(status, STATUS_OK);
            assert_eq!(value, 0);

            jamtrack_oc_sort_drop(handle);
        }
    }

    #[test]
    fn test_oc_sort_tracker_count_initial() {
        unsafe {
            let handle = jamtrack_oc_sort_create(0.5);
            assert!(!handle.is_null());

            let mut value: usize = 999;
            let status = jamtrack_oc_sort_tracker_count(handle, &mut value);
            assert_eq!(status, STATUS_OK);
            assert_eq!(value, 0);

            jamtrack_oc_sort_drop(handle);
        }
    }

    #[test]
    fn test_oc_sort_accessors_after_update() {
        unsafe {
            let handle = jamtrack_oc_sort_create(0.5);
            assert!(!handle.is_null());

            let objects = [make_test_cobject(10.0, 20.0, 100.0, 200.0, 0.9)];
            let mut out = CObjectArray {
                data: std::ptr::null(),
                length: 0,
                _priv: std::ptr::null_mut(),
            };

            let status =
                jamtrack_oc_sort_update(handle, objects.as_ptr(), objects.len(), &mut out);
            assert_eq!(status, STATUS_OK);
            jamtrack_object_array_drop(&mut out);

            let mut fc: usize = 0;
            let status = jamtrack_oc_sort_frame_count(handle, &mut fc);
            assert_eq!(status, STATUS_OK);
            assert_eq!(fc, 1);

            let mut tc: usize = 0;
            let status = jamtrack_oc_sort_tracker_count(handle, &mut tc);
            assert_eq!(status, STATUS_OK);
            assert!(tc > 0);

            jamtrack_oc_sort_drop(handle);
        }
    }

    #[test]
    fn test_oc_sort_accessor_null_handle() {
        unsafe {
            let mut value: usize = 0;
            assert_eq!(
                jamtrack_oc_sort_frame_count(std::ptr::null_mut(), &mut value),
                STATUS_NULL_POINTER
            );
            assert_eq!(
                jamtrack_oc_sort_tracker_count(std::ptr::null_mut(), &mut value),
                STATUS_NULL_POINTER
            );
        }
    }

    #[test]
    fn test_oc_sort_accessor_null_out() {
        unsafe {
            let handle = jamtrack_oc_sort_create(0.5);
            assert!(!handle.is_null());

            assert_eq!(
                jamtrack_oc_sort_frame_count(handle, std::ptr::null_mut()),
                STATUS_NULL_POINTER
            );
            assert_eq!(
                jamtrack_oc_sort_tracker_count(handle, std::ptr::null_mut()),
                STATUS_NULL_POINTER
            );

            jamtrack_oc_sort_drop(handle);
        }
    }

    // -----------------------------------------------------------------------
    // BoostTracker create_with_config + accessors
    // -----------------------------------------------------------------------

    #[test]
    fn test_boost_tracker_create_with_config() {
        unsafe {
            let handle = jamtrack_boost_tracker_create_with_config(
                0.5, 0.3, 30, 3, 0.5, 0.25, 0.25, true, true, false, false, false,
            );
            assert!(!handle.is_null());
            jamtrack_boost_tracker_drop(handle);
        }
    }

    #[test]
    fn test_boost_tracker_create_with_boost_plus() {
        unsafe {
            let handle = jamtrack_boost_tracker_create_with_config(
                0.5, 0.3, 30, 3, 0.5, 0.25, 0.25, true, true, true, false, false,
            );
            assert!(!handle.is_null());
            jamtrack_boost_tracker_drop(handle);
        }
    }

    #[test]
    fn test_boost_tracker_create_with_boost_plus_plus() {
        unsafe {
            // Both plus and plus_plus true — plus_plus should take priority
            let handle = jamtrack_boost_tracker_create_with_config(
                0.5, 0.3, 30, 3, 0.5, 0.25, 0.25, true, true, true, true, false,
            );
            assert!(!handle.is_null());

            // Verify ++ mode is active (soft_boost + varying_threshold)
            let tracker = &*(handle as *const BoostTracker);
            assert!(tracker.uses_soft_boost(), "plus_plus should enable soft_boost");
            assert!(
                tracker.uses_varying_threshold(),
                "plus_plus should enable varying_threshold"
            );

            jamtrack_boost_tracker_drop(handle);
        }
    }

    #[test]
    fn test_boost_tracker_plus_does_not_enable_plus_plus() {
        unsafe {
            // Only plus true — should NOT enable soft_boost / varying_threshold
            let handle = jamtrack_boost_tracker_create_with_config(
                0.5, 0.3, 30, 3, 0.5, 0.25, 0.25, true, true, true, false, false,
            );
            assert!(!handle.is_null());

            let tracker = &*(handle as *const BoostTracker);
            assert!(!tracker.uses_soft_boost(), "plus should not enable soft_boost");
            assert!(
                !tracker.uses_varying_threshold(),
                "plus should not enable varying_threshold"
            );

            jamtrack_boost_tracker_drop(handle);
        }
    }

    #[test]
    fn test_boost_tracker_frame_count_initial() {
        unsafe {
            let handle = jamtrack_boost_tracker_create(0.5, 0.3, 30, 3);
            assert!(!handle.is_null());

            let mut value: usize = 999;
            let status = jamtrack_boost_tracker_frame_count(handle, &mut value);
            assert_eq!(status, STATUS_OK);
            assert_eq!(value, 0);

            jamtrack_boost_tracker_drop(handle);
        }
    }

    #[test]
    fn test_boost_tracker_tracker_count_initial() {
        unsafe {
            let handle = jamtrack_boost_tracker_create(0.5, 0.3, 30, 3);
            assert!(!handle.is_null());

            let mut value: usize = 999;
            let status = jamtrack_boost_tracker_tracker_count(handle, &mut value);
            assert_eq!(status, STATUS_OK);
            assert_eq!(value, 0);

            jamtrack_boost_tracker_drop(handle);
        }
    }

    #[test]
    fn test_boost_tracker_accessors_after_update() {
        unsafe {
            let handle = jamtrack_boost_tracker_create(0.5, 0.3, 30, 3);
            assert!(!handle.is_null());

            let objects = [make_test_cobject(10.0, 20.0, 100.0, 200.0, 0.9)];
            let mut out = CObjectArray {
                data: std::ptr::null(),
                length: 0,
                _priv: std::ptr::null_mut(),
            };

            let status = jamtrack_boost_tracker_update(
                handle,
                objects.as_ptr(),
                objects.len(),
                &mut out,
            );
            assert_eq!(status, STATUS_OK);
            jamtrack_object_array_drop(&mut out);

            let mut fc: usize = 0;
            let status = jamtrack_boost_tracker_frame_count(handle, &mut fc);
            assert_eq!(status, STATUS_OK);
            assert_eq!(fc, 1);

            let mut tc: usize = 0;
            let status = jamtrack_boost_tracker_tracker_count(handle, &mut tc);
            assert_eq!(status, STATUS_OK);
            assert!(tc > 0);

            jamtrack_boost_tracker_drop(handle);
        }
    }

    #[test]
    fn test_boost_tracker_accessor_null_handle() {
        unsafe {
            let mut value: usize = 0;
            assert_eq!(
                jamtrack_boost_tracker_frame_count(std::ptr::null_mut(), &mut value),
                STATUS_NULL_POINTER
            );
            assert_eq!(
                jamtrack_boost_tracker_tracker_count(std::ptr::null_mut(), &mut value),
                STATUS_NULL_POINTER
            );
        }
    }

    #[test]
    fn test_boost_tracker_accessor_null_out() {
        unsafe {
            let handle = jamtrack_boost_tracker_create(0.5, 0.3, 30, 3);
            assert!(!handle.is_null());

            assert_eq!(
                jamtrack_boost_tracker_frame_count(handle, std::ptr::null_mut()),
                STATUS_NULL_POINTER
            );
            assert_eq!(
                jamtrack_boost_tracker_tracker_count(handle, std::ptr::null_mut()),
                STATUS_NULL_POINTER
            );

            jamtrack_boost_tracker_drop(handle);
        }
    }
}

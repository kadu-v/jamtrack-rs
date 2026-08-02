#ifndef JAMTRACK_H
#define JAMTRACK_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -----------------------------------------------------------------------
 * Status codes
 * ----------------------------------------------------------------------- */

#define JAMTRACK_STATUS_OK             0
#define JAMTRACK_STATUS_NULL_POINTER   1
#define JAMTRACK_STATUS_INVALID_ARG    2
#define JAMTRACK_STATUS_INTERNAL_ERROR 3

/* -----------------------------------------------------------------------
 * Data types
 * ----------------------------------------------------------------------- */

typedef struct {
    float x;
    float y;
    float width;
    float height;
    float prob;
    int32_t track_id;   /* -1 = no track assigned */
} CObject;

typedef struct {
    const CObject *data;
    size_t length;
    void *_priv;        /* internal – do not touch */
} CObjectArray;

typedef struct {
    float x;
    float y;
} CFastTrackerPoint;

typedef struct {
    CFastTrackerPoint e1;
    CFastTrackerPoint e2;
    CFastTrackerPoint o2;
    CFastTrackerPoint o1;
} CFastTrackerRoi;

/* -----------------------------------------------------------------------
 * ByteTracker
 * ----------------------------------------------------------------------- */

void *jamtrack_byte_tracker_create(
    size_t frame_rate,
    size_t track_buffer,
    float track_thresh,
    float high_thresh,
    float match_thresh
);

int32_t jamtrack_byte_tracker_update(
    void *handle,
    const CObject *objects,
    size_t length,
    CObjectArray *out_array
);

void jamtrack_byte_tracker_drop(void *handle);

/* -----------------------------------------------------------------------
 * OCSort
 * ----------------------------------------------------------------------- */

void *jamtrack_oc_sort_create(float det_thresh);

void *jamtrack_oc_sort_create_with_config(
    float det_thresh,
    size_t max_age,
    size_t min_hits,
    float iou_threshold,
    size_t delta_t,
    float inertia,
    bool use_byte
);

int32_t jamtrack_oc_sort_update(
    void *handle,
    const CObject *objects,
    size_t length,
    CObjectArray *out_array
);

int32_t jamtrack_oc_sort_frame_count(void *handle, size_t *out_value);
int32_t jamtrack_oc_sort_tracker_count(void *handle, size_t *out_value);

void jamtrack_oc_sort_drop(void *handle);

/* -----------------------------------------------------------------------
 * BoostTracker
 * ----------------------------------------------------------------------- */

void *jamtrack_boost_tracker_create(
    float det_thresh,
    float iou_threshold,
    size_t max_age,
    size_t min_hits
);

void *jamtrack_boost_tracker_create_with_config(
    float det_thresh,
    float iou_threshold,
    size_t max_age,
    size_t min_hits,
    float lambda_iou,
    float lambda_mhd,
    float lambda_shape,
    bool use_dlo_boost,
    bool use_duo_boost,
    bool enable_boost_plus,
    bool enable_boost_plus_plus,
    bool use_shape_similarity_v1
);

int32_t jamtrack_boost_tracker_update(
    void *handle,
    const CObject *objects,
    size_t length,
    CObjectArray *out_array
);

int32_t jamtrack_boost_tracker_frame_count(void *handle, size_t *out_value);
int32_t jamtrack_boost_tracker_tracker_count(void *handle, size_t *out_value);

void jamtrack_boost_tracker_drop(void *handle);

/* -----------------------------------------------------------------------
 * FastTracker
 * ----------------------------------------------------------------------- */

void *jamtrack_fast_tracker_create(
    size_t frame_rate,
    size_t track_buffer,
    float track_thresh,
    float match_thresh
);

void *jamtrack_fast_tracker_create_with_config(
    size_t frame_rate,
    size_t track_buffer,
    float track_thresh,
    float match_thresh,
    size_t reset_velocity_offset,
    size_t reset_position_offset,
    float enlarge_bbox,
    float dampen_motion,
    size_t active_occlusion_to_lost,
    float init_iou_suppression,
    const CFastTrackerRoi *rois,
    size_t roi_count,
    size_t roi_repair_max_gap,
    size_t direction_window,
    float direction_margin_degrees,
    bool mot20
);

int32_t jamtrack_fast_tracker_update(
    void *handle,
    const CObject *objects,
    size_t length,
    CObjectArray *out_array
);

int32_t jamtrack_fast_tracker_frame_count(void *handle, size_t *out_value);
int32_t jamtrack_fast_tracker_tracker_count(void *handle, size_t *out_value);

void jamtrack_fast_tracker_drop(void *handle);

/* -----------------------------------------------------------------------
 * BotSort
 * ----------------------------------------------------------------------- */

void *jamtrack_bot_sort_create(
    size_t frame_rate,
    size_t track_buffer,
    float track_high_thresh,
    float track_low_thresh,
    float new_track_thresh,
    float match_thresh
);

void *jamtrack_bot_sort_create_with_config(
    size_t frame_rate,
    size_t track_buffer,
    float track_high_thresh,
    float track_low_thresh,
    float new_track_thresh,
    float match_thresh,
    bool use_reid,
    float proximity_thresh,
    float appearance_thresh,
    bool mot20,
    bool use_ecc
);

int32_t jamtrack_bot_sort_update(
    void *handle,
    const CObject *objects,
    size_t length,
    CObjectArray *out_array
);

int32_t jamtrack_bot_sort_update_with_features(
    void *handle,
    const CObject *objects,
    size_t length,
    const float *features,
    size_t feature_dim,
    CObjectArray *out_array
);

int32_t jamtrack_bot_sort_frame_count(void *handle, size_t *out_value);
int32_t jamtrack_bot_sort_tracker_count(void *handle, size_t *out_value);

void jamtrack_bot_sort_drop(void *handle);

/* -----------------------------------------------------------------------
 * Shared
 * ----------------------------------------------------------------------- */

void jamtrack_object_array_drop(CObjectArray *array);

#ifdef __cplusplus
}
#endif

#endif /* JAMTRACK_H */

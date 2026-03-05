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
 * Shared
 * ----------------------------------------------------------------------- */

void jamtrack_object_array_drop(CObjectArray *array);

#ifdef __cplusplus
}
#endif

#endif /* JAMTRACK_H */

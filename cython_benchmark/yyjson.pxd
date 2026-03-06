# Cython declarations for yyjson C library
# https://github.com/ibireme/yyjson

from libc.stddef cimport size_t
from libc.stdint cimport uint8_t

cdef extern from "yyjson.h":
    # Opaque types
    ctypedef struct yyjson_doc:
        pass
    ctypedef struct yyjson_val:
        pass
    ctypedef struct yyjson_arr_iter:
        size_t idx
        size_t max
        yyjson_val *cur

    # Read flags
    ctypedef unsigned int yyjson_read_flag
    yyjson_read_flag YYJSON_READ_NOFLAG

    # Document API
    yyjson_doc *yyjson_read(const char *dat, size_t len,
                            yyjson_read_flag flg)
    void yyjson_doc_free(yyjson_doc *doc)
    yyjson_val *yyjson_doc_get_root(yyjson_doc *doc)

    # Value getters
    const char *yyjson_get_str(yyjson_val *val)
    int yyjson_get_int(yyjson_val *val)
    double yyjson_get_real(yyjson_val *val)
    bint yyjson_is_null(yyjson_val *val)

    # Array iteration
    size_t yyjson_arr_size(yyjson_val *arr)
    bint yyjson_arr_iter_init(yyjson_val *arr, yyjson_arr_iter *iter)
    yyjson_arr_iter yyjson_arr_iter_with(yyjson_val *arr)
    bint yyjson_arr_iter_has_next(yyjson_arr_iter *iter)
    yyjson_val *yyjson_arr_iter_next(yyjson_arr_iter *iter)

    # Object field access
    yyjson_val *yyjson_obj_get(yyjson_val *obj, const char *key)
    yyjson_val *yyjson_obj_getn(yyjson_val *obj, const char *key, size_t len)

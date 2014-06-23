/* ************************************************************************** */
/* ************************************************************************** */
/*                                                                            */
/* Author: Jarret Shook                                                       */
/*                                                                            */
/* Module: utils.c                                                            */
/*                                                                            */
/* Modifications:                                                             */
/*                                                                            */
/* 17-June-4: Version 1.0: Created                                            */
/* 17-Jun-14: Version 1.0: Last Updated                                       */
/*                                                                            */
/* Timeperiod: ev8                                                            */
/*                                                                            */
/* ************************************************************************** */
/* ************************************************************************** */

#include "utils.h"

/* ************************************************************************** */
/* ************************************************************************** */

void quick_sort_helper(size_t* arr, size_t* buffer, size_t* start, size_t* end)
{
    int offset;
    size_t* index, *buffer_index, *middle;

    if (start + 1 == end || end - 1 == start || start == end) return;

    middle = start + ((end - start) / 2);

    buffer_index = buffer;

    offset = 0;

    for (index = start + 1; index != end + 1; ++index)
    {
        if (*(index - 1) <= *middle && index - 1 != middle) *buffer_index++ = *(index - 1);

        if (index != end + 1 && *index < *(index - 1)) ++offset;

    }

    if (offset == 0) return;

    offset = buffer_index - buffer;

    *buffer_index++ = *middle;

    for (index = start; index != end + 1; ++index)
    {
        if (*index > *middle) *buffer_index++ = *index;

    }

    //for (index = start; index != end + 1; ++index) printf("%d ", *index);

    //printf("\n");

    memcpy(start, buffer, ((end - start) + 1) * sizeof(size_t));

    //for (index = start; index != end + 1; ++index) printf("%d ", *index);

    //printf("\n");

    middle = start + offset;

    quick_sort_helper(arr, buffer, start, middle);
    quick_sort_helper(arr, buffer, middle, end);

}

void quick_sort(size_t* arr, size_t* start, size_t* end)
{
    size_t* buffer;

    buffer = (size_t*)malloc(sizeof(size_t)* ((end - start) + 1));

    quick_sort_helper(arr, buffer, start, end);

    free(buffer);

}

size_t* binary_search(size_t* arr, size_t* start, size_t* end, size_t key)
{
    size_t* middle;

    if (start == end || start == end - 1)
    {
        if (*start == key || *end == key) return start;

        else return NULL;

    }

    middle = start + ((end - start) / 2);

    if (*middle > key) return binary_search(arr, start, middle, key);

    else return binary_search(arr, middle, end, key);

}

size_t* set_up_arr(size_t* arr)
{
    size_t* new_arr;
    size_t count, counting;
    size_t size;

    int i, index;

    counting = i = 0;

    size = arr[0];

    assert(arr);

    count = 1;

    if (size > 1)
    {
        count = 1;

        for (i = 3; i < size + 2; ++i)
        {
            if (arr[i] != arr[i - 1]) ++count;

        }

    }

    counting = count + 2;

    new_arr = (size_t*)malloc(sizeof(size_t) * ((count * 2) + 2));

    new_arr[0] = count;
    new_arr[1] = size;
    new_arr[2] = arr[2];

    memset(new_arr + 3, 0, sizeof(size_t) * ((count * 2) - 1));

    index = 3;

    for (i = 3; i < size + 2; ++i)
    {
        if (arr[i] != arr[i - 1])
        {
            new_arr[index++] = arr[i];

            ++new_arr[counting++];

        }

        else
        {
            ++new_arr[counting];

        }

    }

    ++new_arr[counting];

    free(arr);

    return new_arr;

}

/* ************************************************************************** */
/* ************************************************************************** */

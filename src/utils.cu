/* ************************************************************************** */
/* ************************************************************************** */
/*                                                                            */
/* Author: Jarret Shook                                                       */
/*                                                                            */
/* Module: utils.cu                                                           */
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

#if __CUDA__

/* ************************************************************************** */
/* ************************************************************************** */

#define SHARED_MEM_SIZE 512

// Handle the Merge.
// Need to make sure there is only one thread operating at each merge.  This means that at least 50% of the threads are not doing anything, and
// that number will increase at each step of the while loop.  Not much we can do about that without overlocking code.

__global__ void merge_sort_gpu(size_t* global_arr, size_t size)
{
    size_t* arr;

    size_t index, offset, count, start, end;

    __shared__ size_t shared_arr[SHARED_MEM_SIZE];

    index = threadIdx.x + blockIdx.x * blockDim.x;
    
    offset = 1;

    arr = global_arr;

    while (offset != SIZE)
    {   
        // Choose the thread at the last index of the first section to merge
        
        if (threadIdx.x % offset == 0) 
        {
            start = 0;
            end = offset;

            // Merge in O(n), backwards
            // index - start = last index of the first section to merge
            // end + offset = last index of the second section to merge

            for (count = 0; count < offset * 2; ++count)
            {
                if ((start < offset && arr[index - start] < arr[index + end]) || end <= 0)
                {
                    shared_arr[(index - start) % SHARED_MEM_SIZE] = arr[index - start++];

                }

                else
                {
                    shared_arr[(index + offset) % SHARED_MEM_SIZE] = arr[index + offset++];

                }

            }

        }

        arr = shared_arr;

        offset *= 2;

        // Potentially dangerous operation if some threads finish and move onto the next section before others
        // Lock needed to avoid that.

        __synchthreads();

    }

    // At this point everything has been merged inside the shared memory
    // Need to merge blocks now.

    global_arr[index] = shared_arr[index % SHARED_MEM_SIZE];

}

/* ************************************************************************** */
/* ************************************************************************** */

#endif /* __CUDA__ */

/* ************************************************************************** */
/* ************************************************************************** */
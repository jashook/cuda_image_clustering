/* ************************************************************************** */
/* ************************************************************************** */
/*                                                                            */
/* Author: Jarret Shook                                                       */
/*                                                                            */
/* Module: read.h                                                             */
/*                                                                            */
/* Modifications:                                                             */
/*                                                                            */
/* 3-May-14: Version 1.0: Created                                             */
/*                                                                            */
/* Timeperiod: ev8                                                            */
/*                                                                            */
/* ************************************************************************** */
/* ************************************************************************** */

#include "mem.h"

/* ************************************************************************** */
/* ************************************************************************** */

void* alloc_host(size_t size)
{
    return malloc(size);

}

void* alloc_device(size_t size)
{
    #ifdef __CUDA__
        void* data;

        cudaError_t error = cudaMalloc(&data, size);
   
        if (error != cudaSuccess)
        {
            printf("Error in cudaMalloc, halting execution\n");

            exit(1);

        }

        return data;

    #else
        return NULL;

    #endif
}

void free_host(void* ptr) 
{
    free(ptr);

}

void free_device(void* ptr)
{
    #ifdef __CUDA__

    cudaError_t error = cudaFree(pointer);
   
    if (error != cudaSuccess)
    {
        printf("Error in cudaFree, halting execution\n");

        exit(1);

     }

}

/* ************************************************************************** */
/* ************************************************************************** */

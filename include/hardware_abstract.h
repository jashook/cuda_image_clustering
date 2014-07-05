/* ************************************************************************** */
/* ************************************************************************** */
/*                                                                            */
/* Author: Jarret Shook                                                       */
/*                                                                            */
/* Module: hardware_abstract.h                                                */
/*                                                                            */
/* Modifications:                                                             */
/*                                                                            */
/* 4-Jul-14: Version 1.0: Created                                             */
/* 4-Jul-14: Version 1.0: Last updated                                        */
/*                                                                            */
/* Timeperiod: ev8n                                                           */
/*                                                                            */
/* ************************************************************************** */
/* ************************************************************************** */

#ifndef __HARDWARE_ABSTRACT_H__
#define __HARDWARE_ABSTRACT_H__

/* ************************************************************************** */
/* ************************************************************************** */

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/* ************************************************************************** */
/* ************************************************************************** */

#define MAX_GPU_COUNT 8

#define GET_LOCKS() s_cuda_create_device_locks()
#define DEV_ID(thread_ptr) thread_ptr->index % s_cuda_gpu_count()

/* ************************************************************************** */
/* ************************************************************************** */


static lock* s_cuda_create_device_locks(void)
{
   static lock s_lock_arr[MAX_GPU_COUNT];

   return &s_lock_arr;
}

static cudaStream* s_cuda_create_device_streams(void)
{
   static cudaStream_t s_stream_arr[MAX_GPU_COUNT];

   return &s_stream_arr;

}

static size_t s_cuda_gpu_count(void)
{
   static int s_device_count = 0;

   cudaDeviceProp dev_prop;

   if (!s_device_count)
   {
      cudaGetDeviceCount(&s_device_count);

   }

   return (size_t)s_device_count;

}

static s_cuda_lock_stream(thread* thread_ptr)
{ 
   lock_get(GET_LOCKS()[DEV_ID(thread_ptr)]);

}

static s_cuda_unlock_stream(thread* thread_ptr)
{
   lock_release(GET_LOCKS()[DEV_ID(thread_ptr)]);

}

/* ************************************************************************** */
/* NOT Thread Safe                                                            */
/* ************************************************************************** */
void cuda_device_abstract_init(void); 

size_t cuda_get_device_id(thread_ptr*);
cudaStream_t* cuda_get_thread_stream(thread*)
void cuda_push_to_stream(thread*, void (*)(void));


/* ************************************************************************** */
/* ************************************************************************** */

#endif /* __HARDWARE_ABSTRACT_H__ */

/* ************************************************************************** */
/* ************************************************************************** */

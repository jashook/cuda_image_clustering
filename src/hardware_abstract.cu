/* ************************************************************************** */
/* ************************************************************************** */
/*                                                                            */
/* Author: Jarret Shook                                                       */
/*                                                                            */
/* Module: hardware_abstract.cu                                               */
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

#include "hardware_abstract.h"

/* ************************************************************************** */
/* ************************************************************************** */

/* ************************************************************************** */
/* NOT Thread Safe                                                            */
/* ************************************************************************** */
void cuda_device_abstract_init(void)
{
   s_cuda_gpu_count();

}

size_t cuda_get_device_id(thread* thread_ptr)
{
   return GET_ID(thread_ptr);

}

cudaStream_t* cuda_get_thread_stream(thread* thread_ptr)
{
   return &s_cuda_create_device_streams()[GET_ID(thread_ptr)];

}

void cuda_push_to_stream(thread* thread_ptr, void (*f_callback)(void))
{
   s_cuda_lock_stream(thread_ptr);

   f_callback();

   s_cuda_unlock_stream(thread_ptr);

}

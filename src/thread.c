/* ************************************************************************** */
/* ************************************************************************** */
/*                                                                            */
/* Author: Jarret Shook                                                       */
/*                                                                            */
/* Module: thread.h                                                           */
/*                                                                            */
/* Modifications:                                                             */
/*                                                                            */
/* 17-June-4: Version 1.0: Created                                            */
/* 17-Jun-14: Version 1.0: Last Updated                                       */
/*                                                                            */
/*                                                                            */
/* Timeperiod: ev8                                                            */
/*                                                                            */
/* ************************************************************************** */
/* ************************************************************************** */

#include "thread.h"

/* ************************************************************************** */
/* ************************************************************************** */

void thread_init(thread* thread_ptr, size_t max_index)
{
    static lock __thread_count_lock;
    static size_t __thread_count_lock_initialized = 0;
    static size_t __thread_index = 0;

    if (!__thread_count_lock_initialized)
    {
        lock_init(&__thread_count_lock);

        __thread_count_lock_initialized = 1;

    }

    #ifdef _WIN32
        memset(thread_ptr, 0, sizeof(thread));

    #else
        memset(thread_ptr, 0, sizeof(thread));

    #endif

    lock_get(&__thread_count_lock);

    ++__thread_index;

    thread_ptr->index = __thread_index;

    lock_release_t(&__thread_count_lock);

    thread_ptr->max_index = max_index;

}

void thread_start(thread* thread_ptr, entry_function_ptr entry_ptr, void* start_arg, size_t stack_size)
{
    #ifdef _WIN32
        size_t thread_id;

        thread_ptr->handle = (HANDLE)_beginthreadex(NULL, stack_size, entry_ptr, start_arg, (unsigned)0, (unsigned*)&thread_id);

    #else
        pthread_create(&thread_ptr->handle, NULL, entry_ptr, start_arg);

    #endif

}

void thread_join(thread* thread_ptr)
{
    #ifdef _WIN32
        WaitForSingleObject(thread_ptr->handle, INFINITE);

    #else
        pthread_join(thread_ptr->handle, NULL);

    #endif
    
}

/* ************************************************************************** */
/* ************************************************************************** */

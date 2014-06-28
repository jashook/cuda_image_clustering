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

#ifndef __THREAD_H__ 
#define __THREAD_H__

/* ************************************************************************** */
/* ************************************************************************** */

#ifdef _WIN32

#include <process.h>
#include <Windows.h>

#else

#include <pthread.h>

#endif

#include <string.h>

#include "lock.h"

/* ************************************************************************** */
/* ************************************************************************** */

#ifdef _WIN32
    typedef unsigned int (__stdcall* entry_function_ptr)(void*);

#else

    typedef void (*entry_function_ptr)(void*);

#endif

typedef struct thread
{
    #ifdef _WIN32
        HANDLE handle;
    #else
        pthread_t handle;
    #endif

    size_t index;
    size_t max_index;

} thread;

/* ************************************************************************** */
/* ************************************************************************** */

void thread_init(thread*, size_t);
void thread_start(thread*, entry_function_ptr, void*, size_t);
void thread_join(thread*);

/* ************************************************************************** */
/* ************************************************************************** */

#endif

/* ************************************************************************** */
/* ************************************************************************** */

/* ************************************************************************** */
/* ************************************************************************** */
/*                                                                            */
/* Author: Jarret Shook                                                       */
/*                                                                            */
/* Module: lock.c                                                             */
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

#include "lock.h"

/* ************************************************************************** */
/* ************************************************************************** */

void lock_delete(lock* lock_ptr)
{
    #ifdef _WIN32
        DeleteCriticalSection(&lock_ptr->mutex);

    #else
        pthread_mutex_destroy(&lock_ptr->mutex);

    #endif

}

void lock_get(lock* lock_ptr)
{
    #ifdef _WIN32
        EnterCriticalSection(&lock_ptr->mutex);

    #else
        pthread_mutex_lock(&lock_ptr->mutex);

    #endif

}

void lock_init(lock* lock_ptr)
{
    #ifdef _WIN32
        InitializeCriticalSection(&lock_ptr->mutex);

    #else
        pthread_mutex_create(&lock_ptr->mutex, NULL);

    #endif

}

void lock_release(lock* lock_ptr)
{
    #ifdef _WIN32
        LeaveCriticalSection(&lock_ptr->mutex);

    #else
        pthread_mutex_unlock(&lock_ptr->mutex);

    #endif
    
}

/* ************************************************************************** */
/* ************************************************************************** */

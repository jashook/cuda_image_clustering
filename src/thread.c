/* ************************************************************************** */
/* ************************************************************************** */
/*                                                                            */
/* Author: Jarret Shook                                                       */
/*                                                                            */
/* Module: thread.h                                                           */
/*                                                                            */
/* Modifications:                                                             */
/*                                                                            */
/* 3-May-14: Version 1.0: Created                                             */
/*                                                                            */
/* Timeperiod: ev8                                                            */
/*                                                                            */
/* ************************************************************************** */
/* ************************************************************************** */

#ifndef __THREAD_H__
#define __THREAD_H__

/* ************************************************************************** */
/* ************************************************************************** */

#include <process.h>
#include <Windows.h>

/* ************************************************************************** */
/* ************************************************************************** */

#define HOST_THREAD 0
#define CUDA_THREAD 1

/* ************************************************************************** */
/* ************************************************************************** */


void thread_join(thread* t)
{
    WaitForSingleObject(t->handle, INFINITE);

}

size_t thread_start(thread* t)
{
    t->handle = (HANDLE)_beginthreadex(NULL, t->stack_size, t->entry_function, (void*)t, (unsigned)0, (unsigned)&t->thread_id);

}

void thread_sleep(thread* t, size_t milliseconds)
{
    Sleep((DWORD)milliseconds);

}

/* ************************************************************************** */
/* ************************************************************************** */

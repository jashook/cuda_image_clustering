/* ************************************************************************** */
/* ************************************************************************** */
/*                                                                            */
/* Author: Jarret Shook                                                       */
/*                                                                            */
/* Module: cluster.h                                                          */
/*                                                                            */
/* Modifications:                                                             */
/*                                                                            */
/* 11-May-14: Version 1.0: Created                                            */
/* 11-May-14: Version 1.0: Last Updated                                       */
/*                                                                            */
/*                                                                            */
/* Timeperiod: ev8                                                            */
/*                                                                            */
/* ************************************************************************** */
/* ************************************************************************** */

#ifndef __CLUSTER_H__
#define __CLUSTER_H__

/* ************************************************************************** */
/* ************************************************************************** */

#ifdef _WIN32

#include <windows.h>

#endif

#include <stdlib.h>
#include <string.h>

#include "lock.h"
#include "picture.h"
#include "thread.h"
#include "thread_arr_arg.h"
#include "utils.h"

/* ************************************************************************** */
/* ************************************************************************** */

typedef struct cluster_index
{
    picture* picture;
    size_t cluster_number;

} cluster_index;

typedef struct cluster_ref_index
{
    size_t cluster_number;
    picture* first_picture_in_cluster;

} cluster_ref_index;

/* ************************************************************************** */
/* ************************************************************************** */

static const double cluster_number = .4;

/* ************************************************************************** */
/* ************************************************************************** */

void build_cluster(picture*, size_t, cluster_index*, cluster_ref_index*);

#ifdef _WIN32
unsigned int __stdcall build_cluster_helper(void*);
#else
void* read_png_files_t_helper(void*);
#endif

void create_first_cluster(picture*, cluster_index*, cluster_ref_index*);
cluster_index* cluster_images(picture*, size_t);
float compare_two_images(size_t*, size_t*);
void init_cluster(cluster_index**, cluster_ref_index**, picture*, size_t);

/* ************************************************************************** */
/* ************************************************************************** */

#endif

/* ************************************************************************** */
/* ************************************************************************** */

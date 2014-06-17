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

void build_cluster(picture*, cluster_index*, cluster_ref_index*);
void create_first_cluster(picture*, cluster_index*, cluster_ref_index*);
cluster_index* cluster_images(picture*, size_t);
float compare_two_images(size_t*, size_t*);
void init_cluster(cluster_index**, cluster_ref_index**, picture*, size_t);

/* ************************************************************************** */
/* ************************************************************************** */

#endif

/* ************************************************************************** */
/* ************************************************************************** */

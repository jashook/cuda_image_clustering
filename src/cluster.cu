/* ************************************************************************** */
/* ************************************************************************** */
/*                                                                            */
/* Author: Jarret Shook                                                       */
/*                                                                            */
/* Module: cluster.c                                                          */
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

#include "cluster.h"

/* ************************************************************************** */
/* ************************************************************************** */

void build_cluster(picture* picture_arr, size_t picture_size, cluster_index* cluster, cluster_index* cluster_ref_table)
{
    size_t cluster_index, index, inner_index, added;

    picture* current_picture;

    cluster_index = 1; /* one cluster has already been added */

    added = 0;

    for (index = 0; index < picture_size; ++index)
    {
        for (inner_index = 0; inner_index < cluster_index; ++inner_cluster)
        {
            current_picture = cluster_ref_table[inner_index].first_picture_in_cluster;

            if (compare_two_images(current_picture->value_arr, picture_arr[index].value_arr) > cluster_number)
            {
                cluster[index].cluster_number = inner_index;
                
                added = 1;

                break;

            }

        }

        if (!added)
        {
            cluster[index].cluster_number = cluster_index;

            cluster_ref_table[cluster_index++] = cluster[index].picture;

        }

    }

}

void create_first_cluster(picture* picture_arr, cluster_index* cluster, cluster_index* cluster_ref_table)
{
    size_t picture_index, cluster_index;

    picture_index = cluster_index = 0;

    /* cluster = final cluster                                       */
    /* cluster_ref_table points to the first picture in each cluster */

    cluster[picture_index].cluster_number = cluster_index;
    cluster_ref_table[cluster_index] = cluster[picture_index].picture;

}

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

lock cluster_lock;

void build_cluster(picture* picture_arr, size_t picture_size, cluster_index* cluster, cluster_ref_index* cluster_ref_table)
{
    size_t cluster_index, index, inner_index, added;

    picture* current_picture;

    cluster_index = 1; /* one cluster has already been added */

    added = 0;

    for (index = 1; index < picture_size; ++index)
    {
        for (inner_index = 0; inner_index < cluster_index; ++inner_index)
        {
            lock_get(&cluster_lock);

            current_picture = cluster_ref_table[inner_index].first_picture_in_cluster;

            lock_release_t(&cluster_lock);

            // avoid a comparison with the same image
            if (current_picture->filename == picture_arr[index].filename) continue;

            if (compare_two_images(current_picture->value_arr, picture_arr[index].value_arr) > cluster_number)
            {
                cluster[index].cluster_number = inner_index;
                
                added = 1;

                break;

            }

        }

        if (!added)
        {
            lock_get(&cluster_lock);

            cluster[index].cluster_number = cluster_index;

            cluster_ref_table[cluster_index++].first_picture_in_cluster = cluster[index].picture;

            lock_release_t(&cluster_lock);

        }

    }

}

#ifdef _WIN32
unsigned int __stdcall build_cluster_helper(void* start_arg)
#else
void* build_cluster_helper(void* start_arg)
#endif
{
    thread_arr_helper* arg;
    size_t picture_size;
    picture* pictures;
    cluster_index* cluster;
    cluster_ref_index* cluster_ref_table;

    arg = (thread_arr_helper*) start_arg;

    pictures = arg->start;
    picture_size = arg->size;
    cluster = arg->cluster;
    cluster_ref_table = arg->cluster_ref_table;

    build_cluster(pictures, picture_size, cluster, cluster_ref_table);

    return 0;
   
}

cluster_index* cluster_images(picture* pictures, size_t picture_size)
{
    cluster_index* cluster;
    cluster_ref_index* cluster_ref_table;
    thread t_arr[4];
    size_t index, thread_count, split;

    thread_count = 4;

    split = picture_size / thread_count;

    lock_init(&cluster_lock);

    init_cluster(&cluster, &cluster_ref_table, pictures, picture_size);
    create_first_cluster(pictures, cluster, cluster_ref_table);

    for (index = 0; index < thread_count; ++index)
    {
        thread_arr_helper* arg = (thread_arr_helper*)malloc(sizeof(thread_arr_helper));

        arg->start = pictures + (index * split);
        arg->size = index == thread_count - 1 ? picture_size - (split * index) : split;
        arg->cluster = cluster;
        arg->cluster_ref_table = cluster_ref_table;

        thread_start(&t_arr[index], build_cluster_helper, arg, 1024);

    }

    for (index = 0; index < thread_count; ++index) thread_join(&t_arr[index]);

    free(cluster_ref_table);

    lock_delete(&cluster_lock);

    return cluster;

}

float compare_two_images(size_t* first_arr, size_t* second_arr)
{
    size_t intersections, total_hashes, index;
    size_t* search;

    intersections = 0;

    total_hashes = first_arr[1] + second_arr[1];

    for (index = 2; index < first_arr[0]; ++index)
    {
        search = binary_search(second_arr + 2, second_arr + 2, second_arr + second_arr[0] + 2, first_arr[index]);

        if (search)
        {
            intersections += first_arr[index + first_arr[0]];

            intersections += second_arr[(search - second_arr) + second_arr[0]];

        }

    }

    return intersections / (float) total_hashes;

}

void create_first_cluster(picture* picture_arr, cluster_index* cluster, cluster_ref_index* cluster_ref_table)
{
    size_t picture_index, cluster_index;

    picture_index = cluster_index = 0;

    /* cluster = final cluster                                       */
    /* cluster_ref_table points to the first picture in each cluster */

    cluster[picture_index].cluster_number = cluster_index;
    cluster_ref_table[cluster_index].first_picture_in_cluster = cluster[picture_index].picture;

}

void init_cluster(cluster_index** cluster, cluster_ref_index** cluster_ref_table, picture* pictures, size_t picture_size)
{
    size_t index;

    *cluster = (cluster_index*)malloc(sizeof(cluster_index) * picture_size);

    *cluster_ref_table = (cluster_ref_index*)malloc(sizeof(cluster_ref_index) * picture_size);

    for (index = 0; index < picture_size; ++index)
    {
        (*cluster)[index].picture = &pictures[index];
        (*cluster)[index].cluster_number = 0;

        (*cluster_ref_table)[index].cluster_number = index;
        (*cluster_ref_table)[index].first_picture_in_cluster = NULL;

    }

    /* all of the data structures used by build cluster are set up by this point */

}

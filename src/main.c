/* ************************************************************************** */
/* ************************************************************************** */
/*                                                                            */
/* Author: Jarret Shook                                                       */
/*                                                                            */
/* Module: main.c                                                             */
/*                                                                            */
/* Modifications:                                                             */
/*                                                                            */
/* 3-May-14: Version 1.0: Created                                             */
/* 3-May-14: Version 1.0: Last Updated                                        */
/*                                                                            */
/*                                                                            */
/* Timeperiod: ev8                                                            */
/*                                                                            */
/* ************************************************************************** */
/* ************************************************************************** */

#ifndef __CUDA__

/* ************************************************************************** */
/* ************************************************************************** */

#include <stdlib.h>
#include <string.h>

#include "cluster.h"
#include "picture.h"
#include "read.h"
#include "utils.h"
#include "vector.h"

/* ************************************************************************** */
/* ************************************************************************** */

char* check_arguments(int argc, char** args, size_t* merged_output)
{
    *merged_output = 0;

    if (argc < 2 || argc > 4)
    {
        printf("Incorrect Usage: exe <csv file> <merged_output:TRUE, FALSE>\n");

        return NULL;

    }

    else 
    {
        if (argc == 3) {
            *merged_output = (args[2][0] == 'T' || args[2][0] == 't') ? 1 : 0; 

        }

        return args[1];

    }

}

/* ************************************************************************** */
/* ************************************************************************** */

#define SIZE 200
#define GPU_THREAD_COUNT 512

int main(int argc, char** args)
{
    size_t merged_output, picture_size;
    picture* picture_table;
    cluster_index* cluster;
    char* filename;
    size_t index;

    picture_table = NULL;

    filename = check_arguments(argc, args, &merged_output);

    if (!filename) return 1;

    if (merged_output) read_csv_file(filename);

    else picture_table = read_txt_file(filename, &picture_size);

    cluster = cluster_images(picture_table, picture_size);

    for (index = 0; index < picture_size; ++index) printf("Picture: %s, Cluster Number: %d\n", cluster[index].picture->filename, (int)cluster[index].cluster_number);

    free(cluster);

    return 0;

}

/* ************************************************************************** */
/* ************************************************************************** */

#endif

/* ************************************************************************** */
/* ************************************************************************** */

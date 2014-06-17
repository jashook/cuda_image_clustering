/* ************************************************************************** */
/* ************************************************************************** */
/*                                                                            */
/* Author: Jarret Shook                                                       */
/*                                                                            */
/* Module: read.cu                                                            */
/*                                                                            */
/* Modifications:                                                             */
/*                                                                            */
/* 3-May-14: Version 1.0: Created                                             */
/*                                                                            */
/* Timeperiod: ev8                                                            */
/*                                                                            */
/* ************************************************************************** */
/* ************************************************************************** */

#include "vector.h"

void read_png_files(void* start_arg)
{
    vector* picture_table;
    size_t start, end;

    picture_table = (vector*)(((void**)start_arg)[0]);
    start = *(size_t*)(((void**)start_arg)[1]);
    end = *(size_t*)(((void**)start_arg)[2]);

    printf("picture_table: %p, start: %d, end: %d", picture_table, start, end);

}

int main()
{
    void* thread_arg;
    size_t file_count;
    vector* picture_table;
    int i;

    file_count = 1000;

    for (i = 0; i < 4; ++i)
    {
        thread_arg = malloc(sizeof(size_t) * 2 + sizeof(vector*));

        ((vector*)thread_arg) = picture_table;
        ((size_t*)(((char*)thread_arg) + sizeof(vector*))) = (file_count / 4) * i;
        ((size_t*)(((char*)thread_arg) + sizeof(vector*)))[1] = (file_count / 4) * i + (file_count / 4);

        read_png_files(thread_arg);

    }

    return 0;
}

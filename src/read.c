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

#include "read.h"

/* ************************************************************************** */
/* ************************************************************************** */

size_t hash(void* number)
{
   return *(size_t*)number;

}

size_t hash_string(unsigned const char* string, size_t size)
{
    size_t hash, i;

    for (hash = i = 0; i < size; ++i)
    {
        hash += string[i];
        hash += (hash << 10);
        hash ^= (hash >> 6);
    }

    hash += (hash << 3);
    hash ^= (hash >> 11);
    hash += (hash << 15);

    return (size_t)abs((int)hash);

}

void hash_picture(const unsigned char* image, size_t height, size_t width, picture* current_picture)
{
    size_t i, size, hash_value;

    size = width * height;

    current_picture->value_arr = (size_t*)malloc(sizeof(size_t) * (((size / 160) * 2) + 2));
   
    for (i = 0; i < (size / 160); ++i)
    {
        hash_value = hash_string(image + (160 * i), (size_t)40 * 40);
      
        current_picture->value_arr[i + 2] = abs((int)hash_value);

    }

    current_picture->value_arr[0] = (size / 160);
    current_picture->value_arr[1] = current_picture->value_arr[0];

}

void read_csv_file(const char* filename)
{
    FILE* file;
    picture* current_picture;
    size_t file_count, index, length, split, thread_count;
    vector* picture_table;
    char* ret_val;
    thread_arr_arg* thread_arg;

    file_count = thread_count = 0;

    picture_table = (vector*)malloc(sizeof(vector));

    vector_init(picture_table, malloc, free);

    char number[10], url[10000], URL[10], path[256], link_path[256];

    file = fopen(filename, "r");

    while (!feof(file))
    {
        if (!(fscanf(file, "\"%[^\"]\",\"%[^\"]\",\"%[^\"]\",\"%[^\"]\",\"%[^\"]\"\n", number, url, URL, path, link_path) > 2))
        {
            fscanf(file, "%[^,],\"%[^\"]\",\"%[^\"]\"\n", URL, path, link_path);
        
        }

        current_picture = (picture*)malloc(sizeof(picture));

        current_picture->filename = (char*)malloc(strlen(path) * sizeof(char));

        strcpy(current_picture->filename, path);

        vector_push_back(picture_table, &current_picture);

        ++file_count;

    }

    thread_count = 4;

    split = picture_table->size / 4;

    for (index = 0; index < thread_count; ++index)
    {
        thread_arg = malloc(sizeof(thread_arr_arg*)); /* have each thread free the thread arg */

        thread_arg->start = picture_table->array + (index * split);
        if (((index * split) + split) < picture_table->size && index == thread_count - 1)
        {
            thread_arg->end = picture_table->array + picture_table->size;

        }

        else if (((index * split) + split) > picture_table->size)
        {
            thread_arg->end = picture_table->array + picture_table->size;

        }

        else
        {
            thread_arg->end = picture_table->array + ((index * split) + split);

        }

        /* read_png_files(thread_arg); */

    }

    /* should join here */

    /* can start building cluster (if we are not producing at the same time as consuming) */

}

void read_png_file(picture* current_picture)
{
    unsigned error;
    unsigned char* image;
    unsigned int width, height;

    /* printf("Trying to open the file: %s\n", current_picture->filename); */

    error = lodepng_decode32_file(&image, &width, &height, current_picture->filename);

    if (error) printf("Error reading the image, %u: %s\nLocation: %s\n", error, lodepng_error_text(error), current_picture->filename);

    else
    {
        //hash_picture(image, height, width, current_picture);

        free(image);

    }

}

void read_png_files(void* start_arg)
{
    thread_arr_arg* arg;
    void** start, **end;
    size_t number, total;

    arg = (thread_arr_arg*)start_arg;

    start = (void**)arg->start;
    end = (void**)arg->end;

    total = end - start;

    while(start != end)
    {
        number = total - (end - start) + 1;

        if (number % 25 == 0) printf("%d of %d\n", number, total);

        read_png_file((picture*)(*start));

        ++start;

    }

}

unsigned int __stdcall read_png_files_t_helper(void* start_arg)
{
    read_png_files(start_arg);

    free(start_arg);

    return 0;

}
       
picture* read_txt_file(const char* filename, size_t* picture_arr_size)
{
    FILE* file;
    picture* current_picture;
    size_t file_count, index, length, split, thread_count;
    vector* picture_table;
    char* ret_val;
    thread_arr_arg* thread_arg;
    picture* picture_arr;
    thread t_arr[4];

    file_count = thread_count = 0;

    picture_table = (vector*)malloc(sizeof(vector));

    vector_init(picture_table, malloc, free);

    char path[256];

    file = fopen(filename, "r");

    while (!feof(file))
    {
        ret_val = fgets(path, 256, file);

        if (!ret_val) break;

        current_picture = (picture*)malloc(sizeof(picture));

        length = strlen(path);

        current_picture->filename = (char*)malloc(length * sizeof(char));

        memcpy(current_picture->filename, path, length - 1);

        current_picture->filename[length - 1] = '\0';

        vector_push_back(picture_table, current_picture);

        ++file_count;

    }

    thread_count = 4;

    split = picture_table->size / 4;

    for (index = 0; index < thread_count; ++index) 
    {
        thread_arg = malloc(sizeof(thread_arr_arg)); /* have each thread free the thread arg */

        thread_arg->start = picture_table->array + (index * split);
        if (((index * split) + split) < picture_table->size && index == thread_count - 1)
        {
            thread_arg->end = picture_table->array + picture_table->size;

        }

        else if (((index * split) + split) > picture_table->size)
        {
            thread_arg->end = picture_table->array + picture_table->size;

        }

        else
        {
            thread_arg->end = picture_table->array + ((index * split) + split);

        }

        thread_init(&t_arr[index]);

        thread_start(&t_arr[index], read_png_files_t_helper, thread_arg, 1024);

    }

    /* should join here */

    /* can start building cluster (if we are not producing at the same time as consuming) */

    for (index = 0; index < thread_count; ++index) thread_join(&t_arr[index]);

    picture_arr = (picture*)malloc(sizeof(picture) * picture_table->size);

    for (index = 0; index < picture_table->size; ++index)
    {
        if (vector_at(picture_table, index))
        {
            memcpy(&picture_arr[index], (picture*)vector_at(picture_table, index), sizeof(picture));

            free(vector_at(picture_table, index));

        }

    }

    *picture_arr_size = picture_table->size;

    free(picture_table);

    return picture_arr;

}

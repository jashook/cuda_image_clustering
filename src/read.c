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

    }

    hash += (hash << 4);
    hash ^= (hash >> 16);

    return (size_t)abs((int)hash);

}

void hash_picture(const unsigned char* image, size_t height, size_t width, picture* current_picture)
{
    size_t i, size, hash_value;

    const size_t png_size = 100;

    size = width * height;

    current_picture->value_arr = (size_t*)malloc(sizeof(size_t)* (((size / png_size) * 2) + 2));
   
    for (i = 0; i < (size / png_size); ++i)
    {
        hash_value = hash_string(image + (png_size * i), png_size);

        current_picture->value_arr[i + 2] = abs((int)hash_value);

    }

    current_picture->value_arr[0] = (size / png_size);
    current_picture->value_arr[1] = current_picture->value_arr[0];

}

void read_csv_file(const char* filename)
{
    FILE* file;
    picture* current_picture;
    size_t file_count, index, split, thread_count;
    vector* picture_table;
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

int read_png_file(picture* current_picture)
{
    unsigned error;
    unsigned char* image;
    unsigned int width, height;

    /* printf("Trying to open the file: %s\n", current_picture->filename); */

    error = lodepng_decode24_file(&image, &width, &height, current_picture->filename);

    if (!error)
    {
        hash_picture(image, height, width, current_picture);

        free(image);

        return 1;

    }

    return 0;

}

void read_png_files(void* start_arg)
{
    thread_arr_arg* arg;
    void** start, **end;
    size_t number, total, read;
    picture* picture_arr;

    arg = (thread_arr_arg*)start_arg;

    start = (void**)arg->start;
    end = (void**)arg->end;
    picture_arr = (picture*)arg->picture_arr;

    total = end - start;

    while(start != end)
    {
        number = total - (end - start) + 1;

        if (number % 25 == 0) printf("%d of %d\n", (int)number, (int)total);

        read = read_png_file((picture*)(*start));

        if (read)
        {
            memcpy(picture_arr, (picture*)(*start), sizeof(picture));

            quick_sort(picture_arr->value_arr + 2, picture_arr->value_arr + 2, picture_arr->value_arr + picture_arr->value_arr[0] + 1);

            //for (i = 0; i < picture_arr->value_arr[0] + 2; ++i) printf("%d ", picture_arr->value_arr[i]);

            //printf("\n");

            picture_arr->value_arr = set_up_arr(picture_arr->value_arr);

            //for (i = 0; i < picture_arr->value_arr[0] * 2 + 2; ++i) printf("%d ", picture_arr->value_arr[i]);

            //printf("\n");

            ++picture_arr;

            free(*start);

        }

        ++start;

    }

}

#ifdef _WIN32
unsigned int __stdcall read_png_files_t_helper(void* start_arg)
#else
void* read_png_files_t_helper(void* start_arg)
#endif
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

    picture_arr = (picture*)malloc(sizeof(picture)* picture_table->size);

    for (index = 0; index < thread_count; ++index) 
    {
        thread_arg = malloc(sizeof(thread_arr_arg)); /* have each thread free the thread arg */

        thread_arg->start = picture_table->array + (index * split);
        thread_arg->picture_arr = picture_arr + (index * split);
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

    *picture_arr_size = picture_table->size;

    free(picture_table);

    return picture_arr;

}

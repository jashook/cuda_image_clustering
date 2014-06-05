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

void hash_picture(const unsigned char* const image, size_t height, size_t width, picture* current_picture)
{
    size_t i, size, hash_value;

    size = width * height;

    current_picture->value_arr = (size_t*)malloc(sizeof(size_t) * (((size / 160) * 2) + 2));
   
    for (i = 0; i < size / 160; ++i)
    {
        hash_value = hash_string(image + (160 * i), (size_t)40 * 40);
      
        current_picture->value_arr[i + 2] = abs(hash_value);

    }

}

void read_csv_file(const char* const filename)
{
    FILE* file;
    picture* current_picture;
    size_t i = 0;
    vector picture_table;

    vector_init(&picture_table);

    char number[10], url[10000], URL[10], path[256], link_path[256];

    file = fopen(filename, "r");

    while (!feof(file))
    {
        if (!(fscanf(file, "\"%[^\"]\",\"%[^\"]\",\"%[^\"]\",\"%[^\"]\",\"%[^\"]\"\n", number, url, URL, path, link_path) > 2))
        {
            fscanf(file, "%[^,],\"%[^\"]\",\"%[^\"]\"\n", URL, path, link_path);
        
        }

        current_picture = (picture*)malloc(sizeof(picture));

        current_picture.filename = (char*)malloc(strlen(path) * sizeof(char));

        strcpy(current_picture.filename, path);

        vector_push_back(&picture_table, &current_picture);

        read_png_file(&current_picture);

    }
}

void read_png_file(picture* current_picture)
{
    unsigned error;
    unsigned char* image;
    unsigned int width, height;

    printf("Trying to open the file: %s\n", filename);

    error = lodepng_decode32_file(&image, &width, &height, filename);

    if (error) printf("Error reading the image, %u: %s\n", error, lodepng_error_text(error));

    else
    {
        hash_picture(image, height, width, current_picture);

        free(image);

    }

}
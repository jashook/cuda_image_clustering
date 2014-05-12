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

#define _CRT_SECURE_NO_DEPRECATE
#define __EV8__ 1

/* ************************************************************************** */
/* ************************************************************************** */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "lodepng.h"
#include "hash_table.h"
#include "vector.h"

/* ************************************************************************** */
/* ************************************************************************** */

char* check_arguments(int, char**);
size_t hash(void*);
size_t hash_string(unsigned const char*, size_t);
void hash_picture(unsigned const char* const, size_t, size_t, hash_table*);
void read_csv_file(const char* const);
void read_png_file(const char* const, hash_table*);

/* ************************************************************************** */
/* ************************************************************************** */

typedef struct picture
{
   unsigned char* image;
   unsigned int height;
   unsigned int width;

} picture;

/* ************************************************************************** */
/* ************************************************************************** */

char* check_arguments(int argc, char** args)
{
   char* file;

   if (argc != 2)
   {
      printf("Error: correct usage <executable> <path to image>\n");
      file = NULL;

   }

   else
   {
      file = args[1];

   }

   return file;

}

void combined_free(void* pointer)
{
   #ifdef __EV8__
   
      free(pointer);
   
   #elif __EV8_CUDA_
      cudaError_t error = cudaFree(pointer);
   
      if (error != cudaSuccess)
      {
         printf("Error in cudaFree, halting execution\n");

         exit(1);

      }
   
   #else
   
      printf("Cannot determine correct free, halting execution\n");

      exit(1);
   
   #endif

}

void* combined_malloc(size_t size)
{
   #ifdef __EV8__
   
      return malloc(size);
   
   #elif __EV8_CUDA__
   
      void* data;
   
      cudaError_t error = cudaMalloc(&data, size);
   
      if (error != cudaSuccess)
      {
         printf("Error in cudaMalloc, halting execution\n");

         exit(1);

      }
   
      return data;
   
   #endif
   
   return NULL;

}

size_t hash(void* number)
{
   return *(int*)number;

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

void hash_picture(const unsigned char* const image, size_t height, size_t width, hash_table* table)
{
   size_t i;
   
   for (i = 0; i < size; ++i)
   {
      hash_value = hash_string(image + (160 * i), (size_t)40 * 40);
      
      hash_table_insert_ignore_collision(table, &hash_value, &hash_value, &hash);

   }

}

void parse_set_init(vector* vec)
{
   size_t i;
   picture* image_arr;
   unsigned char* images_host, images_device;

   size_t size, current_index;

   current_index = 0;

   size = 0;

   cudaMalloc((void**)&image_arr, vec->size * sizeof(picture));

   for (i = 0; i < vec->size; ++i) size += ((picture*)vec->array[i])->height * ((picture*)vec->array[i])->width;

   images_host = (unsigned char*)malloc(size * sizeof(unsigned char));

   for (i = 0; i < vec->size; ++i) 
   {
      memcpy(images_host + current_index, ((picture*)vec->array[i])->image, sizeof(unsigned char) * (((picture*)vec->array[i])->height * ((picture*)vec->array[i])->width)); 
   
      current_index += ((picture*)vec->array[i])->height * ((picture*)vec->array[i])->width;

   }

   cudaMalloc((void**)&images_device, size * sizeof(unsigned char));

   cudaMemcpy((void**)&images_device, images_host, size, cudaMemcpyHostToDevice);

   cudaMemcpy(&image_arr, vec->array, vec->size * sizeof(picture), cudaMemcpyHostToDevice); 
}

void read_csv_file(const char* const filename)
{
   FILE* file;
   hash_table table;
   size_t i = 0;

   char number[10], url[10000], URL[10], path[256], link_path[256];

   file = fopen(filename, "r");

   hash_table_init(&table);

   hash_table_reserve(&table, 10000000, &combined_malloc);

   while (!feof(file))
   {
      /*if (i == 20)
      {
         parse_set_init(&vec);


      }*/

      if (fscanf(file, "\"%[^\"]\",\"%[^\"]\",\"%[^\"]\",\"%[^\"]\",\"%[^\"]\"\n", number, url, URL, path, link_path) > 2) 
      {
         read_png_file(path, &table);

      }

      else 
      {
         fscanf(file, "%[^,],\"%[^\"]\",\"%[^\"]\"\n", URL, path, link_path);
         
         read_png_file(path, &table);

      }

   }


}

void read_png_file(const char* const filename, hash_table* table)
{
   unsigned error;
   unsigned char* image;
   unsigned char** image_d;
   unsigned int width, height;

   picture* pic;

   printf("Trying to open the file: %s\n", filename);

   error = lodepng_decode32_file(&image, &width, &height, filename);

   if (error) printf("Error reading the image, %u: %s\n", error, lodepng_error_text(error));

   else
   {
      cudaMalloc(image_d, (width * height) * sizeof(unsigned char));

      cudaMemcpy(image_d, image, (width * height) * sizeof(unsigned char), cudaMemcpyHostToDevice);

      hash_picture(*image_d, height, width, table);

      free(image);

   }

}

void test_hash_table(const char* const filename)
{

   hash_table table;
   int i;
   
   hash_table_init(&table);
   
   hash_table_reserve(&table, 10000, combined_malloc);
   
   printf("-----\nHash Table, reserve\n-----\ntable:%p\nmax_size:%d\nsize:%d\ncollisions:%d\n", table.array, table.max_size, table.size, table.collisions);
   
   i = 0;
   
   printf ("Searching for value: %d\nReturns:%d values\n",i, hash_table_search(&table, &i, &hash)->collisions);
   
   hash_table_clear(&table);
   
   printf ("Searching for value: %d\nReturns:%d\n",i, hash_table_search(&table, &i, &hash)->collisions);
   
   hash_table_free(&table, &combined_free);

}

void test_vector()
{
   size_t arr[10], arr2[10];
   size_t i, j;
   vector vec;

   j = 200;

   vector_init(&vec, &malloc, &free);

   for (i = 0; i < 10; ++i)
   {
      arr[i] = i;

      vector_push_back(&vec, &arr[i]);

   }

   for (i = 0; i < 10; ++i)
   {
      arr2[i] = i + 10;

      vector_push_back(&vec, &arr2[i]);

   }

   vector_insert(&vec, &j, vec.size);
   vector_remove(&vec, 0);

   printf("-----\nVector testing\n-----\nSize: %d\nMax size: %d\nContents: ", vector_size(&vec), vector_max_size(&vec));

   for (i = 0; i < 20; ++i) printf("%d ", *(size_t*)vector_pop_back(&vec));

   vector_push_back(&vec, &j);

   printf("\nFront: %d\nBack: %d\n", *(size_t*)vector_front(&vec), *(size_t*)vector_back(&vec));

   vector_clear(&vec);

   printf("Vector cleared\nSize: %d\n", vec.size);

   vector_free(&vec);

   printf("\n");

}

/* ************************************************************************** */
/* ************************************************************************** */

int main(int argc, char** args)
{
   char* filename = check_arguments(argc, args);

   if (!filename) return 1;

   read_csv_file(filename);

   return 0;

}

/* ************************************************************************** */
/* ************************************************************************** */

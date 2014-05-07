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
#define __STD__ 1

/* ************************************************************************** */
/* ************************************************************************** */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
   #ifdef __STD__
   
      free(pointer);
   
   #elif __CUDA__
   
      void* data;
   
      cudaError_t error = cudaFree(pointer);
   
      if (error != cudaSuccess)
      {
         printf("Error in cudaFree, halting execution\n");

         exit(1);

      }
   
      return data;
   
   #else
   
      printf("Cannot determine correct free, halting execution\n");

      exit(1);
   
   #endif

}

void* combined_malloc(size_t size)
{
   #ifdef __STD__
   
      return malloc(size);
   
   #elif __CUDA__
   
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
   size_t hash_value;

   size_t* value = (size_t*)malloc(sizeof(size_t));

   for (i = 0; i < (height * width) / 160; ++i)
   {
      hash_value = *value = hash_string(image + (160 * i), (size_t)40 * 40);
      
      //hash_table_insert(table, &hash_value, (void*)value, &hash, &combined_malloc);
   
   }

}

void read_csv_file(const char* const filename)
{
   FILE* file;
   vector vec;
   unsigned char* image;
   size_t i = 0;

   char number[10], url[10000], URL[10], path[256], link_path[256];

   file = fopen(filename, "r");

   vector_init(&vec, &malloc, &free);

   while (!feof(file))
   {
      if (fscanf(file, "\"%[^\"]\",\"%[^\"]\",\"%[^\"]\",\"%[^\"]\",\"%[^\"]\"\n", number, url, URL, path, link_path) > 2) 
      {
         read_png_file(path, &vec);

      }

      else 
      {
         fscanf(file, "%[^,],\"%[^\"]\",\"%[^\"]\"\n", URL, path, link_path);
         
         read_png_file(path, &vec);

      }

   }

   printf("Total hashes: %d\n", vector_size(&vec));

}

void read_png_file(const char* const filename, vector* vec)
{
   unsigned error;
   unsigned char* image;
   unsigned int width, height;

   picture* pic;

   printf("Trying to open the file: %s\n", filename);

   error = lodepng_decode32_file(&image, &width, &height, filename);

   if (error) printf("Error reading the image, %u: %s\n", error, lodepng_error_text(error));

   else
   {
      pic = (picture*)malloc(sizeof(picture));

      pic->image = image;
      pic->width = width;
      pic->height = height;

      vector_push_back(vec, pic);

   }

}

void test_hash_table(const char* const filename)
{

   hash_table table;
   int i;
   
   hash_table_init(&table);
   
   hash_table_reserve(&table, 10000, combined_malloc);
   
   printf("-----\nHash Table, reserve\n-----\ntable:%p\nmax_size:%d\nsize:%d\ncollisions:%d\n", table.array, table.max_size, table.size, table.collisions);

   read_png_file(filename, &table);
   
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

   return 0;

}

/* ************************************************************************** */
/* ************************************************************************** */

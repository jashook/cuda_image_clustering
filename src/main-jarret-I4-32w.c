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

/* ************************************************************************** */
/* ************************************************************************** */

char* check_arguments(int, char**);
size_t hash(int*);
size_t hash_string(unsigned const char*, size_t);
void hash_picture(unsigned const char* const, size_t, size_t, hash_table*);
void read_csv_file(const char* const);
void read_png_file(const char* const, hash_table*);

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

size_t hash(int* number)
{
   return *number;

}

size_t hash_string(unsigned const char* string, size_t size)
{
    int hash, i;

    for (hash = i = 0; i < size; ++i)
    {
        hash += string[i];
        hash += (hash << 10);
        hash ^= (hash >> 6);
    }

    hash += (hash << 3);
    hash ^= (hash >> 11);
    hash += (hash << 15);

    return (size_t)abs(hash);

}

void hash_picture(const unsigned char* const image, size_t height, size_t width, hash_table* table)
{
   size_t i;
   size_t hash_value;
   
   size_t* value = (size_t*)malloc(sizeof(size_t));

   for (i = 0; i < (height * width) / 160; ++i)
   {
      hash_value = *value = hash_string((void*)(image + (160 * i)), (size_t)40 * 40);
      
      hash_table_insert(table, &hash_value, value, &hash, &combined_malloc);
   
   }

}

void read_csv_file(const char* const filename)
{
   FILE* file;
   char number[10];
   char url[1000];
   char URL[1000];
   char path[256];
   char link_path[256];



   file = fopen(filename, "r");

   while (!feof(file))
   {
      fscanf(file, "%[^','],%[^','],%[^','],%[^'s'],%s", number, url, URL, path, link_path);

      //read_png_file(path);

   }

}

void read_png_file(const char* const filename, hash_table* table)
{
   unsigned error;
   unsigned char* image;
   unsigned int width, height;

   printf("Trying to open the file: %s\n", filename);

   error = lodepng_decode32_file(&image, &width, &height, filename);

   if (error) printf("Error reading the image, %u: %s\n", error, lodepng_error_text(error));

   else
   {
      hash_picture(image, height, width, table);

      free(image);

   }

}

void test_hash_table(const char* const filename)
{

   hash_table table;

   int i;
   
   hash_table_init(&table);
   
   hash_table_reserve(&table, 1000000, combined_malloc);
   
   printf("-----\nHash Table, reserve\n-----\ntable:%p\nmax_size:%d\nsize:%d\ncollisions:%d\n", table.array, table.max_size, table.size, table.collisions);

   read_png_file(filename, &table);
   
   i = 0;
   
   printf ("Searching for value: %d\nReturns:%d values\n",i, hash_table_search(&table, &i, &hash)->collisions);
   
   hash_table_clear(&table);
   
   printf ("Searching for value: %d\nReturns:%d\n",i, hash_table_search(&table, &i, &hash)->collisions);
   
   hash_table_free(&table, &combined_free);

}

/* ************************************************************************** */
/* ************************************************************************** */

int main(int argc, char** args)
{
   char* filename = check_arguments(argc, args);

   if (!filename) return 1;

   test_hash_table(filename);
   
   
   return 0;

}

/* ************************************************************************** */
/* ************************************************************************** */

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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lodepng.h"

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

size_t hash(unsigned const char* string, size_t size)
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

    return hash;

}

void hash_picture(unsigned const char* const image, size_t width, size_t height)
{
   size_t i;

   for (i = 0; i < (height * width) / 160; ++i) printf("%d\n", hash(image + (160 * i), (size_t)40 * 40));

}

void read_png_file(const char* const filename)
{
   unsigned error;
   unsigned char* image;
   size_t width, height;
   size_t i = 0;

   printf("Trying to open the file: %s\n", filename);

   error = lodepng_decode32_file(&image, &width, &height, filename);

   if (error) printf("Error reading the image, %u: %s\n", error, lodepng_error_text(error));

   else
   {
      hash_picture(image, height, width);

      free(image);

   }

}

/* ************************************************************************** */
/* ************************************************************************** */

int main(int argc, char** args)
{
   char* filename = check_arguments(argc, args);

   if (!filename) return 1;

   read_png_file(filename);

   return 0;

}

/* ************************************************************************** */
/* ************************************************************************** */

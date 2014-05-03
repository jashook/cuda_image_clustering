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

/* ************************************************************************** */
/* ************************************************************************** */

int main(int argc, char** args)
{
   char* filename = check_arguments(argc, args);

   if (filename == NULL) return 1;

   unsigned error;
   unsigned char* image;
   unsigned width, height;
   size_t i = 0;

   printf("Trying to open the file: %s\n", filename);

   error = lodepng_decode32_file(&image, &width, &height, filename);

   if (error) printf("Error reading the image, %u: %s\n", error, lodepng_error_text(error));

   else free(image);

   return 0;

}

/* ************************************************************************** */
/* ************************************************************************** */

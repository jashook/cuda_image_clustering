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

//#define __CUDA__

/* ************************************************************************** */
/* ************************************************************************** */

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "read.h"
#include "vector.h"

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

   if (!filename) return 1;

   read_csv_file(filename);

   return 0;

}

/* ************************************************************************** */
/* ************************************************************************** */

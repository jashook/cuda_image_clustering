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

/* ************************************************************************** */
/* ************************************************************************** */

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

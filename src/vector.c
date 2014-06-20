/* ************************************************************************** */
/* ************************************************************************** */
/*                                                                            */
/* Author: Jarret Shook                                                       */
/*                                                                            */
/* Module: vector.c                                                           */
/*                                                                            */
/* Modifications:                                                             */
/*                                                                            */
/* 5-May-14: Version 1.0: Created                                             */
/* 5-May-14: Version 1.0: Last Updated                                        */
/*                                                                            */
/*                                                                            */
/* Timeperiod: ev8                                                            */
/*                                                                            */
/* ************************************************************************** */
/* ************************************************************************** */

#include "vector.h"

/* ************************************************************************** */
/* Constant Variables                                                         */
/* ************************************************************************** */

const size_t VECTOR_BASE_SIZE = 512;

/* ************************************************************************** */
/* General Function for a vector                                              */
/* ************************************************************************** */

/* ************************************************************************** */
/* ************************************************************************** */
/* Function: at                                                               */
/*                                                                            */
/* Arguments: vector*, size_t index                                           */
/*                                                                            */
/* Return: void*                                                               */
/*                                                                            */
/* ************************************************************************** */
/* ************************************************************************** */

void* vector_at(vector* vec, size_t index)
{
   return index > vec->size ? NULL : vec->array[index];
   
}

/* ************************************************************************** */
/* ************************************************************************** */
/* Function: back                                                             */
/*                                                                            */
/* Arguments: vector*                                                         */
/*                                                                            */
/* Return: void*: last object in vector                                       */
/*                                                                            */
/* ************************************************************************** */
/* ************************************************************************** */

void* vector_back(vector* vec)
{
   return vec->array[vec->size - 1];

}

/* ************************************************************************** */
/* ************************************************************************** */
/* Function: clear                                                            */
/*                                                                            */
/* Arguments: vector*                                                         */
/*                                                                            */
/* Return: void                                                               */
/*                                                                            */
/* ************************************************************************** */
/* ************************************************************************** */

void vector_clear(vector* vec)
{
   memset(vec->array, 0, sizeof(void*) * vec->max_size);

   vec->size = 0;

}

/* ************************************************************************** */
/* ************************************************************************** */
/* Function: free                                                             */
/*                                                                            */
/* Arguments: vector*                                                         */
/*                                                                            */
/* Return: void                                                               */
/*                                                                            */
/* ************************************************************************** */
/* ************************************************************************** */

void vector_free(vector* vec)
{
   vec->vector_free(vec->array);

}

/* ************************************************************************** */
/* ************************************************************************** */
/* Function: front                                                            */
/*                                                                            */
/* Arguments: vector*                                                         */
/*                                                                            */
/* Return: void*: pointer to object at the first index                        */
/*                                                                            */
/* ************************************************************************** */
/* ************************************************************************** */

void* vector_front(vector* vec)
{
   return vec->array[0];

}

/* ************************************************************************** */
/* ************************************************************************** */
/* Function: init                                                             */
/*                                                                            */
/* Arguments: vector*, pointer to malloc, pointer to free                     */
/*                                                                            */
/* Return: void                                                               */
/*                                                                            */
/* ************************************************************************** */
/* ************************************************************************** */

void vector_init(vector* vec, void* (*malloc_ptr)(size_t), void (*free_ptr)(void*))
{
   memset(vec, 0, sizeof(vector));

   vec->vector_malloc = malloc_ptr;
   vec->vector_free = free_ptr;
   
   vec->array = (void**)vec->vector_malloc(sizeof(void*) * VECTOR_BASE_SIZE);

   vec->max_size = VECTOR_BASE_SIZE;

}

/* ************************************************************************** */
/* ************************************************************************** */
/* Function: insert                                                           */
/*                                                                            */
/* Arguments: vector*, void*: data, size_t: index                             */
/*                                                                            */
/* Return: void                                                               */
/*                                                                            */
/* ************************************************************************** */
/* ************************************************************************** */

void vector_insert(vector* vec, void* data, size_t index)
{
   size_t i;

   if (vec->size + 1 > vec->max_size) vector_reserve(vec, vec->max_size * 2);
   
   for (i = vec->size - 1; i > index; --i)
   {
      vec->array[i] = vec->array[i - 1];
   
   }
   
   vec->array[index] = data;
   
   ++vec->size;

}

/* ************************************************************************** */
/* ************************************************************************** */
/* Function: max_size                                                         */
/*                                                                            */
/* Arguments: size_t: max size the vector can have                            */
/*                                                                            */
/* Return: void                                                               */
/*                                                                            */
/* ************************************************************************** */
/* ************************************************************************** */

size_t vector_max_size(vector* vec)
{
   return vec->max_size;

}

/* ************************************************************************** */
/* ************************************************************************** */
/* Function: pop_back                                                         */
/*                                                                            */
/* Arguments: vector*                                                         */
/*                                                                            */
/* Return: void*: data                                                        */
/*                                                                            */
/* ************************************************************************** */
/* ************************************************************************** */

void* vector_pop_back(vector* vec)
{
   return vec->array[--vec->size];

}

/* ************************************************************************** */
/* ************************************************************************** */
/* Function: push_back                                                        */
/*                                                                            */
/* Arguments: vector*, void*: data                                            */
/*                                                                            */
/* Return: void                                                               */
/*                                                                            */
/* ************************************************************************** */
/* ************************************************************************** */

void vector_push_back(vector* vec, void* data)
{
   if (vec->size + 1 == vec->max_size) vector_reserve(vec, vec->max_size * 2);

   vec->array[vec->size++] = data;

}

/* ************************************************************************** */
/* ************************************************************************** */
/* Function: remove                                                           */
/*                                                                            */
/* Arguments: vector*, size_t: index                                          */
/*                                                                            */
/* Return: void                                                               */
/*                                                                            */
/* ************************************************************************** */
/* ************************************************************************** */

void vector_remove(vector* vec, size_t index)
{
   size_t i;

   for (i = index; i < vec->size; ++i)
   {
      vec->array[i] = vec->array[i + 1];
   
   }

   --vec->size;

}

/* ************************************************************************** */
/* ************************************************************************** */
/* Function: reserve                                                          */
/*                                                                            */
/* Arguments: vector*, size_t: index                                          */
/*                                                                            */
/* Return: void                                                               */
/*                                                                            */
/* ************************************************************************** */
/* ************************************************************************** */

void vector_reserve(vector* vec, size_t new_size)
{
    void** temp;

    temp = (void**)vec->vector_malloc(sizeof(void*) * new_size);

    memcpy(temp, vec->array, sizeof(void*) * vec->size);

    vec->vector_free(vec->array);

    vec->array = temp;

}

/* ************************************************************************** */
/* ************************************************************************** */
/* Function: size                                                             */
/*                                                                            */
/* Arguments: vector*                                                         */
/*                                                                            */
/* Return: size: size                                                         */
/*                                                                            */
/* ************************************************************************** */
/* ************************************************************************** */

size_t vector_size(vector* vec)
{
   return vec->size;

}

/* ************************************************************************** */
/* ************************************************************************** */
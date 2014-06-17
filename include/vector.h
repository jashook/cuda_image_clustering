/* ************************************************************************** */
/* ************************************************************************** */
/*                                                                            */
/* Author: Jarret Shook                                                       */
/*                                                                            */
/* Module: vector.h                                                           */
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

#ifndef __VECTOR_H__
#define __VECTOR_H__

/* ************************************************************************** */
/* ************************************************************************** */

#include <stdlib.h>
#include <string.h>

/* ************************************************************************** */
/* ************************************************************************** */

typedef struct vector
{
   void** array;
   size_t max_size;
   size_t size;
   void* (*vector_malloc)(size_t);
   void (*vector_free)(void*);

} vector;

extern const size_t VECTOR_BASE_SIZE;

/* ************************************************************************** */
/* General Function for a vector                                              */
/* ************************************************************************** */

void* vector_at(vector*, size_t);
void* vector_back(vector*);
void vector_clear(vector*);
void vector_free(vector*);
void* vector_front(vector*);
void vector_init(vector*, void* (*malloc_ptr)(size_t), void (*free_ptr)(void*));
void vector_insert(vector*, void*, size_t);
size_t vector_max_size(vector*);
void* vector_pop_back(vector*);
void vector_push_back(vector*, void*);
void vector_remove(vector*, size_t);
void vector_reserve(vector*, size_t);
size_t vector_size(vector*);

/* ************************************************************************** */
/* ************************************************************************** */

#endif /* __HASH_TABLE_H__ */

/* ************************************************************************** */
/* ************************************************************************** */
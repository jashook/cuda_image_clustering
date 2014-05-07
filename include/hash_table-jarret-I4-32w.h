/* ************************************************************************** */
/* ************************************************************************** */
/*                                                                            */
/* Author: Jarret Shook                                                       */
/*                                                                            */
/* Module: hash_table.h                                                       */
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

#ifndef __HASH_TABLE_H__
#define __HASH_TABLE_H__

/* ************************************************************************** */
/* ************************************************************************** */

#include <stdlib.h>
#include <string.h>

/* ************************************************************************** */
/* ************************************************************************** */

typedef struct hash_table_bucket
{
   void* m_data;
   int collisions;
   struct hash_table_bucket* next;

} hash_table_bucket;

typedef struct hash_table
{
   hash_table_bucket* array;
   size_t collisions;
   size_t max_size;
   size_t size;

} hash_table;

/* ************************************************************************** */
/* General Function for a Hash Table                                          */
/* ************************************************************************** */

void hash_table_clear(hash_table*);
void hash_table_free(hash_table*, void (*memory_management)(void*));
void hash_table_init(hash_table*);
int hash_table_insert(hash_table*, void*, void*, size_t (*hash_function)(void*), void* (*memory_management)(size_t));
void hash_table_reserve(hash_table*, size_t, void* (*memory_management)(size_t));
hash_table_bucket* hash_table_search(hash_table*, void*, size_t (*hash_function)(void*));

/* ************************************************************************** */
/* ************************************************************************** */

#endif /* __HASH_TABLE_H__ */

/* ************************************************************************** */
/* ************************************************************************** */
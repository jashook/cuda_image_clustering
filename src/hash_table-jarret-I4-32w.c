/* ************************************************************************** */
/* ************************************************************************** */
/*                                                                            */
/* Author: Jarret Shook                                                       */
/*                                                                            */
/* Module: hash_table.c                                                       */
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

#include "hash_table.h"

/* ************************************************************************** */
/* General Function for a Hash Table                                          */
/* ************************************************************************** */

/* ************************************************************************** */
/* ************************************************************************** */
/* Function: clear                                                            */
/*                                                                            */
/* Arguments: hash_table*                                                     */
/*                                                                            */
/* Return: void                                                               */
/*                                                                            */
/* ************************************************************************** */
/* ************************************************************************** */

void hash_table_clear(hash_table* table)
{
   if (table->array != NULL)
   {
      memset(table->array, 0, sizeof(hash_table_bucket) * table->max_size);
      
      table->size = 0;
      table->collisions = 0;
   
   }

}

/* ************************************************************************** */
/* ************************************************************************** */
/* Function: free                                                             */
/*                                                                            */
/* Arguments: hash_table*, free                                               */
/*                                                                            */
/* Return: void                                                               */
/*                                                                            */
/* ************************************************************************** */
/* ************************************************************************** */

void hash_table_free(hash_table* table, void (*memory_management)(void*))
{
   size_t i;

   for (i = 0; i < table->max_size; ++i)
   {
      hash_table_bucket* temp = table->array[i].next;
   
      while (table->array[i].next != NULL)
      {
         temp = temp->next;
         memory_management(table->array[i].next);
         
         table->array[i].next = temp;
      
      }
   }

}

/* ************************************************************************** */
/* ************************************************************************** */
/* Function: init                                                             */
/*                                                                            */
/* Arguments: hash_table*                                                     */
/*                                                                            */
/* Return: void                                                               */
/*                                                                            */
/* ************************************************************************** */
/* ************************************************************************** */

void hash_table_init(hash_table* table)
{
   memset(table, 0, sizeof(hash_table));

}

/* ************************************************************************** */
/* ************************************************************************** */
/* Function: insert                                                           */
/*                                                                            */
/* Arguments: hash_table*, void* key, void* data, hash_function, malloc       */
/*                                                                            */
/* Return: int: Error Code                                                    */
/*                                                                            */
/* 0: inserts without problems                                                */
/* 1: inserts with collision                                                  */
/*                                                                            */
/* ************************************************************************** */
/* ************************************************************************** */

int hash_table_insert(hash_table* table, void* key, void* data, size_t (*hash_function)(void*), void* (*memory_management)(size_t))
{
   int index = hash_function(key);
   
   if (table->array[index % table->max_size].m_data == NULL)
   {
      table->array[index % table->max_size].m_data = data;
   
   
   }
   
   else
   {
      table->array[index % table->max_size].next = memory_management(sizeof(hash_table_bucket));
      
      memset(table->array[index % table->max_size].next, 0, sizeof(hash_table_bucket));
      
      ++table->collisions;
      
      ++table->array[index % table->max_size].collisions;
      
      return 1;
   
   }
   
   return 0;

}

/* ************************************************************************** */
/* ************************************************************************** */
/* Function: reserve                                                          */
/*                                                                            */
/* Arguments: hash_table*, size_t size, malloc                                */
/*                                                                            */
/* Return:  void                                                              */
/*                                                                            */
/* Notes: assumes hash_table has been initialized                             */
/*                                                                            */
/* ************************************************************************** */
/* ************************************************************************** */

void hash_table_reserve(hash_table* table, size_t size, void* (*memory_management)(size_t))
{
   table->array = memory_management(sizeof(hash_table_bucket) * size);

   memset(table->array, 0, sizeof(hash_table_bucket) * size);
   
   table->max_size = size;

}

/* ************************************************************************** */
/* ************************************************************************** */
/* Function: search                                                           */
/*                                                                            */
/* Arguments: hash_table*, void* key, void* data, hash_function               */
/*                                                                            */
/* Return: hash_table_bucket* pointer to bucket                               */
/*                                                                            */
/* ************************************************************************** */
/* ************************************************************************** */

hash_table_bucket* hash_table_search(hash_table* table, void* key, size_t (*hash_function)(void*))
{
   int index = hash_function(key);
   
   return &table->array[index % table->max_size];

}

/* ************************************************************************** */
/* ************************************************************************** */
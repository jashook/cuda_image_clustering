/* *************************************************************************** */
/* *************************************************************************** */
/*                                                                             */
/* Author: Jarret Shook                                                        */
/*                                                                             */
/* Module: test.cu                                                             */
/*                                                                             */
/* Modifications:                                                              */
/*                                                                             */
/* 5-June-14: Version 1.0: Created                                             */
/*                                                                             */
/* Timeperiod: ev8                                                             */
/*                                                                             */
/* *************************************************************************** */
/* *************************************************************************** */

#include "test.h"

/* ************************************************************************** */
/* ************************************************************************** */

void test_hash_table(const char* const filename)
{

   hash_table table;
   int i;
   
   hash_table_init(&table);
   
   hash_table_reserve(&table, 10000, &malloc);
   
   printf("-----\nHash Table, reserve\n-----\ntable:%p\nmax_size:%d\nsize:%d\ncollisions:%d\n", table.array, table.max_size, table.size, table.collisions);

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


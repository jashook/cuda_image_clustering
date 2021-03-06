/* ************************************************************************** */
/* ************************************************************************** */
/*                                                                            */
/* Author: Jarret Shook                                                       */
/*                                                                            */
/* Module: read.h                                                             */
/*                                                                            */
/* Modifications:                                                             */
/*                                                                            */
/* 3-May-14: Version 1.0: Created                                             */
/*                                                                            */
/* Timeperiod: ev8                                                            */
/*                                                                            */
/* ************************************************************************** */
/* ************************************************************************** */

#ifndef __READ_H__
#define __READ_H__

/* ************************************************************************** */
/* ************************************************************************** */

#include <stdio.h>

#include "lodepng.h"
#include "picture.h"
#include "thread.h"
#include "thread_arr_arg.h"
#include "utils.h"
#include "vector.h"

/* ************************************************************************** */
/* ************************************************************************** */

size_t hash(void*);
size_t hash_string(unsigned const char*, size_t);
void hash_picture(const unsigned char*, size_t, size_t, picture*);
void read_csv_file(const char*);
int read_png_file(picture*);
void read_png_files(void*);

#ifdef _WIN32
unsigned int __stdcall read_png_files_t_helper(void*);
#else
void* read_png_files_t_helper(void*);
#endif

picture* read_txt_file(const char*, size_t*);

/* ************************************************************************** */
/* ************************************************************************** */

#endif

/* ************************************************************************** */
/* ************************************************************************** */


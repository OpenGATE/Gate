#ifndef num_sort_h
#define num_sort_h
#if defined(__STDC__) || defined(__cplusplus)
#if defined(__cplusplus)
extern "C" {
#endif
int compare_short(const void *a, const void *b);
int compare_int(const void *a, const void *b);
int sort_short(short *v, int count);
int sort_int(int *v, int count);
#if defined(__cplusplus)
}
#endif
#else
int compare_short();
int compare_int();
int sort_int();
int sort_short();
#endif
#endif

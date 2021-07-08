#include <stdlib.h>
#if defined(__STDC__) || defined(__cplusplus)
int compare_short(const void *a, const void *b)
#else
int compare_short(a,b)
void *a, *b;
#endif
{
	short sa = *(short*)a, sb = *(short*)b;
    if (sa < sb) return(-1);
    else if (sa > sb) return (1);
    else return (0);
}
#if defined(__STDC__) || defined(__cplusplus)
int compare_int(const void *a, const void *b)
#else
int compare_int(a,b)
void *a, *b;
#endif
{
	int ia = *(int*)a, ib = *(int*)b;
    if (ia < ib) return(-1);
    else if (ia > ib) return (1);
    else return (0);
}

#if defined(__STDC__) || defined(__cplusplus)
int sort_short(short *v, int count)
#else
int sort_short(v,count)
short *v;
int count;
#endif
{
    qsort(v, count, sizeof(short), compare_short);
    return 1;
}

#if defined(__STDC__) || defined(__cplusplus)
int sort_int(int *v, int count)
#else
int sort_int(v,count)
int *v;
int count;
#endif
{
    qsort(v, count, sizeof(int), compare_int);
    return 1;
}

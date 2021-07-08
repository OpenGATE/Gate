/* @(#)scdef.h	1.1 7/11/92 */
/* scdef.h -- used on host for defining ueptable_ offsets,
                  used on SuperCard for creating ueptable_  
*/


/* this section creates the structure data type for the ueptable */

#define sc_stack_option(var) /* stack option = var */ 
#define sc_heap_option(var)  /* heap  option = var */ 
#define scssladdr(name) void (*name)();
#define scuserfun(name) void (*name)();
struct xl_uep_t
	{
	void (*uep_0)();
#include "scssl.h"
	void (*uep_1)();
#include "scfun.h"
	void (*uep_2)();
	};
#undef scssladdr
#undef scuserfun
#undef sc_stack_option
#undef sc_heap_option

/* a specific instance of the structure is created, in order to
   allow the definition of the "x_f" macro. This macro is used in
   a user's host C program to calculate the offset of a given entry
   into the ueptable
*/
struct xl_uep_t xl_uept;
#define x_f(f)  (long) (&xl_uept.uep_1 - &xl_uept.f)

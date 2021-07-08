/*-------------------------------------------------------

           List Mode Format 
                        
     --  poweri8.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of poweri8.c:

     returns a**b
-------------------------------------------------------*/
#include <stdio.h>
#include "lmf.h"
u16 poweri8(u8 a, u8 b)
{

  int i;
  u16 buf;
  buf = (u16) a;
  for (i = 1; i < b; i++)
    buf = buf * ((u16) a);
  return (buf);
}

/* main() */
/* { */
/*   printf("\n2**3 = %d\n\n",poweri8(2,8)); */

/* } */

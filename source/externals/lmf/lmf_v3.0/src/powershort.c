/*-------------------------------------------------------

           List Mode Format 
                        
     --  poweri16.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of poweri16.c:


-------------------------------------------------------*/
#include <stdio.h>
#include "lmf.h"
u16 poweri16(u16 a, u8 b)
{

  int i;
  u16 buf = a;
  for (i = 1; i < b; i++)
    buf = buf * a;
  return (buf);
}

/* main() */
/* { */
/*   printf("\n2**3 = %d\n\n",poweri16(10,4)); */

/* } */

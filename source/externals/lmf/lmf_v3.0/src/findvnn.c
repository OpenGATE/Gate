/*-------------------------------------------------------

           List Mode Format 
                        
     --  findvnn.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of findvnn.c:
     Returns the number of neighbours for event record
     according to neighbouring order.
     The parameter is the encoding event header 
     (or event pattern)

-------------------------------------------------------*/
#include <stdio.h>

#include "lmf.h"
u16 findvnn(u16 ep2)
{
  u16 nn = 0, vnn = 0;
  nn = (ep2 >> 6);
  nn = (nn & 3);		/* extract the order of neighbourhood */
  if (nn == 0)
    vnn = 0;
  if (nn == 1)
    vnn = 4;
  if (nn == 2)
    vnn = 8;
  if (nn == 3)
    vnn = 20;
  return (vnn);
}

u16 findvnn2(u16 nn)
{				/* already extract */
  u16 vnn = 0;

  if (nn == 0)
    vnn = 0;
  if (nn == 1)
    vnn = 4;
  if (nn == 2)
    vnn = 8;
  if (nn == 3)
    vnn = 20;
  return (vnn);
}

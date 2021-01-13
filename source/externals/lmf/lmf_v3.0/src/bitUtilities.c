/*-------------------------------------------------------

List Mode Format 
                        
--  bitUtilities.c  --                      

Martin.Rey@epfl.ch

Crystal Clear Collaboration
Copyright (C) 2004 LPHE/EPFL, CH-1015 Lausanne

This software is distributed under the terms 
of the GNU Lesser General 
Public Licence (LGPL)
See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

Description of bitUtilities

-------------------------------------------------------*/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "lmf.h"


/************************************************

Description of putBits:

return a bit pattern like 00...111 with n 1 if 
nbOfBitsToPut = n (i.e return 2^n-1)

************************************************/

u64 putBits(u8 nbOfBitsToPut)
{
  u64 nb = 0;
  u8 i;


  if (nbOfBitsToPut > 64) {
    printf("Impossible to add more bits than 64 -> exit\n");
    exit(0);
  }

  for (i = 0; i < nbOfBitsToPut; i++)
    nb |= 1 << i;

  return nb;
}

u16 getBitsNb(u16 number)
{
  double tmp;
  u16 quotient;
  double remainder;
  u16 bitsNb;

  tmp = log(number) / log(2);

  quotient = (u16) (tmp);

  remainder = tmp - quotient;

  if ((quotient > 0) && (remainder == 0))
    bitsNb = tmp;
  else
    bitsNb = tmp + 1;

  return bitsNb;
}

u32 poweru32(u8 a, u8 b)
{
  u8 i;
  u32 value, buf;

  buf = (u32) a;
  value = buf;
  for (i = 1; i < b; i++)
    buf *= value;
  return (buf);
}

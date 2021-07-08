/*-------------------------------------------------------

           List Mode Format 
                        
     --  testYourSystemForLMF.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of testYourSystemForLMF.c:
     Test your format size

-------------------------------------------------------*/

#include <stdio.h>
#include "lmf.h"



int testYourSystemForLMF()
{
  if (sizeof(u8) != 1) {
    printf
	("Incompatibility of your system for LMF library libLMF.a \n: size of i8 is %d byte instead of 1\n",
	 sizeof(i8));
    exit(0);
  }

  if (sizeof(u16) != 2) {
    printf
	("Incompatibility of your system for LMF library libLMF.a \n: size of i16 is %d byte instead of 2\n",
	 sizeof(i16));
    exit(0);
  }


  if (sizeof(u32) != 4) {
    printf
	("Incompatibility of your system for LMF library libLMF.a \n: size of int is %d byte instead of 4\n",
	 sizeof(i32));
    exit(0);
  }




  if (sizeof(u64) != 8) {
    printf
	("Incompatibility of your system for LMF library libLMF.a \n: size of i64 is %d byte instead of 8\n",
	 sizeof(i64));
    exit(0);
  }





  return (0);
}

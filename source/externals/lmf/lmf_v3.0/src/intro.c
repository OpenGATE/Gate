/*-------------------------------------------------------

           List Mode Format 
                        
     --  intro.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of intro.c:
     Test your system (size of each format) and display
     introducing message


-------------------------------------------------------*/

#include <stdio.h>
#include "lmf.h"
void intro()
{


  testYourSystemForLMF();
  printf("\n\n");
  printf("******************************************************\n");
  printf("*                                                    *\n");
  printf("*                                                    *\n");
  printf("*               L M F                                *\n");
  printf("*             Release %d.%d                            *\n",
	 VERSION_LMF, SUBVERSION_LMF);
  printf("*                                                    *\n");
  printf("*     Magalie.Krieguer@iphe.unil.ch                  *\n");
  printf("*       Luc.Simon@iphe.unil.ch                       *\n");
  printf("*                                                    *\n");
  printf("*     Crystal Clear Collaboration                    *\n");
  printf("*       Copyright (C) 2003 IPHE/UNIL,                     *\n");
  printf("*          CH-1015 Lausanne                          *\n");
  printf("*                                                    *\n");
  printf("* This software is distributed under the terms       *\n");
  printf("* of the GNU Lesser General Public Licence (LGPL)    *\n");
  printf("* See LMF/LICENSE.txt for further details            *\n");
  printf("*                                                    *\n");
  printf("*                                                    *\n");
  printf("******************************************************\n");
  printf("\n\n\n");


}

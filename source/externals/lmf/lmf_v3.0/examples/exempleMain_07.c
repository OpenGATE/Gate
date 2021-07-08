/*-------------------------------------------------------

           List Mode Format 
                        
     --  exempleMain_05.c  --                      

     Magalie.krieguer@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/
/*------------------------------------------------------------------------

			   
	 Description : 
	 
         How to extract an information from .cch file
	 Example : read and display ring diameter.

---------------------------------------------------------------------------*/

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "../includes/lmf.h"
#include <time.h>


int cch_index = 0;


int main()
{

  contentLMFdata structRingDiameter = { 0 };

  if (LMFcchReader(""))
    exit(EXIT_FAILURE);		/* read file.cch and fill in structures LMF_cch */

  structRingDiameter = getLMF_cchNumericalValue("ring diameter");

  printf("ring diameter-> %f\t%s\n", structRingDiameter.numericalValue,
	 structRingDiameter.unit);

  LMFcchReaderDestructor();

  if (LMFcchReader(""))
    exit(EXIT_FAILURE);		/* read file.cch and fill in structures LMF_cch */

  structRingDiameter = getLMF_cchNumericalValue("ring diameter");

  printf("ring diameter-> %f\t%s\n", structRingDiameter.numericalValue,
	 structRingDiameter.unit);

  LMFcchReaderDestructor();

  printf("main over\n");

  return (EXIT_SUCCESS);


}

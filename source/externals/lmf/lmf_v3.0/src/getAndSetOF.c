/*-------------------------------------------------------

           List Mode Format 
                        
     --  getAndSetOF.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of getAndSetOF.c:
     For developpers only.
     It allows to sent some infos in an ascii file
     like number of singles, ... 


-------------------------------------------------------*/



#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "../includes/lmf.h"



static int gasoDoneOnce = FALSE;
static int gastoDoneOnce = FALSE;
static i8 outputFile[81];
static FILE *pfo = NULL;



int OF_is_Set()
{

  return (gasoDoneOnce || gastoDoneOnce);
}

FILE *getAndSetThisOutputFileName(i8 * name)
{

  if (gastoDoneOnce == FALSE) {

    strcpy(outputFile, name);
    printf("output file = %s\n", outputFile);
    pfo = fopen(outputFile, "w");
    gasoDoneOnce = TRUE;
    gastoDoneOnce = TRUE;
  }

  return (pfo);

}

FILE *getAndSetOutputFileName()
{

  if (gasoDoneOnce == FALSE) {
    printf("Enter output file name :\n");
    scanf("%[^\n]", outputFile);

    pfo = fopen(outputFile, "w");
    gasoDoneOnce = TRUE;
    gastoDoneOnce = TRUE;


  }

  return (pfo);

}

void destroyGetAndSetOutputFileName()
{
  gasoDoneOnce = FALSE;
  gastoDoneOnce = FALSE;

  if (pfo)
    fclose(pfo);
}

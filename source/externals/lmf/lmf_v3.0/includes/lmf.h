/*-------------------------------------------------------

           List Mode Format 
                        
     --  lmf.h  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of lmf.h:
	 general header for LMF library.


-------------------------------------------------------*/
#ifdef __cplusplus
extern "C" {
#endif


#include "lmf_format.h"

#ifndef _LMF_CCS_H
#define _LMF_CCS_H

#ifndef __LMFBI_H
#define __LMFBI_H
#include "structure_LMF.h"	/* LMF Record carriers structures declaration */
#endif

#ifndef __SYMBOLIC_CONSTAN_H
#define __SYMBOLIC_CONSTAN_H
#include "constantsLMF_ccs.h"	/* symbolic constant list for .ccs part */
#include "constantsLMF_cch.h"	/* symbolic constant list for .cch part */
#include "constantsLocateEvents.h"	/* symbolic constant list for calculation of events position */
#endif

#ifndef __PROTOTYPE_H
#define __PROTOTYPE_H

#include "dlist.h"		/* doubly-linked list management    */
#include "prototypesLMF_cch.h"	/* function prototypes for .cch part */
#include "prototypesLMF_ccs.h"	/* function prototypes for .ccs part */
#include "prototypesLocateEvents.h"	/*function prototypes for calculation of events position */

#endif


#endif

#ifdef __cplusplus
}
#endif

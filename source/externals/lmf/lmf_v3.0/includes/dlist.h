/*-------------------------------------------------------

           List Mode Format 
                        
     --  dlist.h  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

    
     Warning : This file is an adaptation of dlist.h found in:
     
     "Mastering Algorithms with C"  by Kyle Loudon,  
     published by O'Reilly & Associates.  This
     code is under copyright and cannot be included 
     in any other book, publication,
     or  educational product  without  
     permission  from  O'Reilly & Associates.


      Description of dlist.h:
	 Declaration of stuctures used to manage
	 the doubly-linked lists.
	 
-------------------------------------------------------*/

/*****************************************************************************
*                                                                            *
*  ------------------------------- dlist.h --------------------------------  *
*                                                                            *
*****************************************************************************/

#ifndef DLIST_H
#define DLIST_H
#include <stdlib.h>
#include "lmf.h"

/*****************************************************************************
*                                                                            *
*  Define a structure for doubly-linked list elements.                       *
*                                                                            *
*****************************************************************************/

typedef struct DListElmt_ {

  void *data;
  struct DListElmt_ *prev;
  struct DListElmt_ *next;
  u8 CWN;			/* coincidence with next if 1 */
} ELEMENT;

typedef ELEMENT *LINK;

/*****************************************************************************
*                                                                            *
*  Define a structure for doubly-linked lists.                               *
*                                                                            *
*****************************************************************************/

typedef struct DList_ {

  int size;

  int (*match) (const void *key1, const void *key2);
  void (*destroy) (void *data);

  ELEMENT *head;
  ELEMENT *tail;

} LIST;

/*****************************************************************************
*                                                                            *
*  --------------------------- Public Interface ---------------------------  *
*                                                                            *
*****************************************************************************/




	      /***    OREILLY Interface    ***/
void dlist_init(LIST * list, void (*destroy) (void *data));

void dlist_destroy(LIST * list);

int dlist_ins_next(LIST * list, ELEMENT * element, const void *data);

int dlist_ins_prev(LIST * list, ELEMENT * element, const void *data);

int dlist_remove(LIST * list, ELEMENT * element, void **data);

#define dlist_size(list) ((list)->size)

#define dlist_head(list) ((list)->head)

#define dlist_tail(list) ((list)->tail)

#define dlist_is_head(element) ((element)->prev == NULL ? 1 : 0)

#define dlist_is_tail(element) ((element)->next == NULL ? 1 : 0)

#define dlist_data(element) ((element)->data)

#define dlist_next(element) ((element)->next)

#define dlist_prev(element) ((element)->prev)

#endif

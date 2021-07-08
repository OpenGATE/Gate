/*-------------------------------------------------------

           List Mode Format 
                        
     --  dlist.c  --                      

     Luc.Simon@iphe.unil.ch

     Crystal Clear Collaboration
     Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

     This software is distributed under the terms 
     of the GNU Lesser General 
     Public Licence (LGPL)
     See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

     Description of dlist.c:


     Warning : This file is an adaptation of dlist.c found in:
     
     "Mastering Algorithms with C"  by Kyle Loudon,  
     published by O'Reilly & Associates.  This
     code is under copyright and cannot be included 
     in any other book, publication,
     or  educational product  without  
     permission  from  O'Reilly & Associates.


      Description of dlist.c:
	 Functions used to manage
	 the doubly-linked lists.
	 
-------------------------------------------------------*/





/*****************************************************************************
*                                                                            *
*  ------------------------------- dlist.c --------------------------------  *
*                                                                            *
*****************************************************************************/
#include <stdio.h>

#include <stdlib.h>

#include <string.h>

#include "lmf.h"

/*****************************************************************************
*                                                                            *
*  ------------------------------ dlist_init ------------------------------  *
*                                                                            *
*****************************************************************************/

void dlist_init(LIST * list, void (*destroy) (void *data))
{

/*****************************************************************************
*                                                                            *
*  Initialize the list.                                                      *
*                                                                            *
*****************************************************************************/

  list->size = 0;
  list->destroy = destroy;
  list->head = NULL;
  list->tail = NULL;

  return;

}

/*****************************************************************************
*                                                                            *
*  ---------------------------- dlist_destroy -----------------------------  *
*                                                                            *
*****************************************************************************/

void dlist_destroy(LIST * list)
{

  void *data;

/*****************************************************************************
*                                                                            *
*  Remove each element.                                                      *
*                                                                            *
*****************************************************************************/

  while (dlist_size(list) > 0) {

    if (dlist_remove(list, dlist_tail(list), (void **) &data) == 0
	&& list->destroy != NULL) {

      /***********************************************************************
      *                                                                      *
      *  Call a user-defined function to free dynamically allocated data.    *
      *                                                                      *
      ***********************************************************************/

      list->destroy(data);

    }

  }

/*****************************************************************************
*                                                                            *
*  No operations are allowed now, but clear the structure as a precaution.   *
*                                                                            *
*****************************************************************************/

  memset(list, 0, sizeof(LIST));

  return;

}

/*****************************************************************************
*                                                                            *
*  ---------------------------- dlist_ins_next ----------------------------  *
*                                                                            *
*****************************************************************************/

int dlist_ins_next(LIST * list, ELEMENT * element, const void *data)
{

  ELEMENT *new_element;

/*****************************************************************************
*                                                                            *
*  Do not allow a NULL element unless the list is empty.                     *
*                                                                            *
*****************************************************************************/

  if (element == NULL && dlist_size(list) != 0)
    return -1;

/*****************************************************************************
*                                                                            *
*  Allocate storage for the element.                                         *
*                                                                            *
*****************************************************************************/

  if ((new_element = (ELEMENT *) malloc(sizeof(ELEMENT))) == NULL)
    return -1;

/* // line added by luc.simon@iphe.unil.ch */
  new_element->CWN = 0;

/*****************************************************************************
*                                                                            *
*  Insert the new element into the list.                                     *
*                                                                            *
*****************************************************************************/

  new_element->data = (void *) data;

  if (dlist_size(list) == 0) {

   /**************************************************************************
   *                                                                         *
   *  Handle insertion when the list is empty.                               *
   *                                                                         *
   **************************************************************************/

    list->head = new_element;
    list->head->prev = NULL;
    list->head->next = NULL;
    list->tail = new_element;

  }

  else {

   /**************************************************************************
   *                                                                         *
   *  Handle insertion when the list is not empty.                           *
   *                                                                         *
   **************************************************************************/

    new_element->next = element->next;
    new_element->prev = element;

    if (element->next == NULL)
      list->tail = new_element;
    else
      element->next->prev = new_element;

    element->next = new_element;

  }

/*****************************************************************************
*                                                                            *
*  Adjust the size of the list to account for the inserted element.          *
*                                                                            *
*****************************************************************************/

  list->size++;

  return 0;

}

/*****************************************************************************
*                                                                            *
*  ---------------------------- dlist_ins_prev ----------------------------  *
*                                                                            *
*****************************************************************************/


int dlist_ins_prev(LIST * list, ELEMENT * element, const void *data)
{

  ELEMENT *new_element;

/*****************************************************************************
*                                                                            *
*  Do not allow a NULL element unless the list is empty.                     *
*                                                                            *
*****************************************************************************/

  if (element == NULL && dlist_size(list) != 0)
    return -1;

/*****************************************************************************
*                                                                            *
*  Allocate storage to be managed by the abstract data type.                 *
*                                                                            *
*****************************************************************************/

  if ((new_element = (ELEMENT *) malloc(sizeof(ELEMENT))) == NULL)
    return -1;


/* // line added by luc.simon@iphe.unil.ch */
  new_element->CWN = 0;


/*****************************************************************************
*                                                                            *
*  Insert the new element into the list.                                     *
*                                                                            *
*****************************************************************************/

  new_element->data = (void *) data;

  if (dlist_size(list) == 0) {

   /**************************************************************************
   *                                                                         *
   *  Handle insertion when the list is empty.                               *
   *                                                                         *
   **************************************************************************/

    list->head = new_element;
    list->head->prev = NULL;
    list->head->next = NULL;
    list->tail = new_element;

  }


  else {

   /**************************************************************************
   *                                                                         *
   *  Handle insertion when the list is not empty.                           *
   *                                                                         *
   **************************************************************************/

    new_element->next = element;
    new_element->prev = element->prev;

    if (element->prev == NULL)
      list->head = new_element;
    else
      element->prev->next = new_element;

    element->prev = new_element;

  }


/*****************************************************************************
*                                                                            *
*  Adjust the size of the list to account for the new element.               *
*                                                                            *
*****************************************************************************/

  list->size++;

  return 0;

}

/*****************************************************************************
*                                                                            *
*  ----------------------------- dlist_remove -----------------------------  *
*                                                                            *
*****************************************************************************/

int dlist_remove(LIST * list, ELEMENT * element, void **data)
{

/*****************************************************************************
*                                                                            *
*  Do not allow a NULL element or removal from an empty list.                *
*                                                                            *
*****************************************************************************/

  if (element == NULL || dlist_size(list) == 0)
    return -1;

/*****************************************************************************
*                                                                            *
*  Remove the element from the list.                                         *
*                                                                            *
*****************************************************************************/

  *data = element->data;

  if (element == list->head) {

   /**************************************************************************
   *                                                                         *
   *  Handle removal from the head of the list.                              *
   *                                                                         *
   **************************************************************************/

    list->head = element->next;

    if (list->head == NULL)
      list->tail = NULL;
    else
      element->next->prev = NULL;

  }

  else {

   /**************************************************************************
   *                                                                         *
   *  Handle removal from other than the head of the list.                   *
   *                                                                         *
   **************************************************************************/

    element->prev->next = element->next;

    if (element->next == NULL)
      list->tail = element->prev;
    else
      element->next->prev = element->prev;

  }

/*****************************************************************************
*                                                                            *
*  Free the storage allocated by the abstract data type.                     *
*                                                                            *
*****************************************************************************/

  free(element);

/*****************************************************************************
*                                                                            *
*  Adjust the size of the list to account for the removed element.           *
*                                                                            *
*****************************************************************************/

  list->size--;

  return 0;

}

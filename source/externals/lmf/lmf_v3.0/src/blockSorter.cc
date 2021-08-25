/*-------------------------------------------------------

List Mode Format 
                        
--  blockSorter.cc  --                      

Martin.Rey@epfl.ch

Crystal Clear Collaboration
Copyright (C) 2005 LPHE/EPFL, CH-1015 Lausanne

This software is distributed under the terms 
of the GNU Lesser General 
Public Licence (LGPL)
See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

Description of blockSorter (version c++)

Allows to re-mix events comming from different sectors.
It is usefull when the events are stored by fixed size block
of specific sectors.
You have to specify the number of sectors you use and 
the size of one block

-------------------------------------------------------*/


#include "timeOrder.hh"

#include <vector>
#include <algorithm>

using namespace std;

static u16 sctNb;
static vector < EVENT_RECORD * >*EventLists;
static u16 *sctName;
static u64 *t;

static CURRENT_CONTENT *pcCC = NULL;	/* to build coinci file head */
static ENCODING_HEADER *pEncoHC = NULL;	/* to build coinci file head */
static EVENT_HEADER *pEHC = NULL;	/* to build coinci file head */
static GATE_DIGI_HEADER *pGDHC = NULL;	/* to build coinci file head */
static COUNT_RATE_HEADER *pCRHC = NULL;	/* to build coinci file head */
static COUNT_RATE_RECORD *pCRRC = NULL;	/* to build coinci file head */
static FILE *pfC = NULL;
static char *fileNameC;

void setNbOfSct(u16 inlineSctNb)
{
  sctNb = inlineSctNb;
}

void copyHeads(const ENCODING_HEADER * pEncoH,
	       const EVENT_HEADER * pEH,
	       const GATE_DIGI_HEADER * pGDH,
	       const COUNT_RATE_HEADER * pCRH,
	       const CURRENT_CONTENT * pcC,
	       const COUNT_RATE_RECORD * pCRR, char *fileNameCCS)
{
  pEncoHC = new ENCODING_HEADER;
  pEHC = new EVENT_HEADER;
  pGDHC = new GATE_DIGI_HEADER;
  pCRHC = new COUNT_RATE_HEADER;
  pcCC = new CURRENT_CONTENT;
  pCRRC = new COUNT_RATE_RECORD;

  if ((pEncoHC == NULL) || (pEHC == NULL) || (pGDHC == NULL)
      || (pCRHC == NULL) || (pcCC == NULL) || (pCRRC == NULL))
    printf("\n *** error : blockSorter.c : copyHeads : malloc\n");

  if (pEncoH)
    *pEncoHC = *pEncoH;		/* no pointer in this structure, it is safe */
  if (pEH)
    *pEHC = *pEH;		/* no pointer in this structure, it is safe */
  if (pCRH)
    *pCRHC = *pCRH;		/*  no pointer in this structure, it is safe */
  if (pGDH)
    *pGDHC = *pGDH;		/*  no pointer in this structure, it is safe */
  if (pcC)
    *pcCC = *pcC;		/*  no pointer in this structure, it is safe */

  fileNameC = fileNameCCS;
}

void sortBlocks(const ENCODING_HEADER * pEncoH,
		const EVENT_HEADER * pEH,
		const COUNT_RATE_HEADER * pCRH,
		const GATE_DIGI_HEADER * pGDH,
		const CURRENT_CONTENT * pcC,
		const EVENT_RECORD * pER,
		const COUNT_RATE_RECORD * pCRR, char *fileNameCCS)
{
  static u32 viewOnce = 0;
  static u8 doneonce = 0;
  static u8 check = 0;
  u8 notEmpty;

  EVENT_RECORD *pERin;
  vector < EVENT_RECORD * >::iterator iter, searcher, destroyer;

  u16 sct, index, old = (u16) - 1;
  u64 tOld;

  if (!doneonce) {
    if (!(EventLists = new vector < EVENT_RECORD * >[sctNb]))
      printf("\n *** error : blockSorter.c : sortBlocks : malloc\n");
    if (!(sctName = new u16[sctNb]))
      printf("\n *** error : blockSorter.c : sortBlocks : malloc\n");
    if (!(t = new u64[sctNb]))
      printf("\n *** error : blockSorter.c : sortBlocks : malloc\n");

    copyHeads(pEncoH, pEH, pGDH, pCRH, pcC, pCRR, fileNameCCS);
    doneonce++;
  }

  pERin = newER(pEHC);		/* complete allocatation for an element */

  copyER(pER, pERin, pEHC);	/* *pERin = *pER but safe */

  sct = getRsectorID(pEncoHC, pERin);

  if (!((viewOnce >> sct) & 1)) {
    viewOnce |= 1 << sct;
    check = 0;
    for (index = 0; index < 8 * sizeof(u32); index++)
      check += (viewOnce >> index) & 1;
    sctName[check - 1] = sct;
    if (check > sctNb) {
      printf
	  ("nb of sector in file is greater than the one introduced\nPlease re-run\n");
      exit(0);
    }
  }

  for (index = 0; index < check; index++)
    if (sct == sctName[index])
      break;

  iter =
      lower_bound(EventLists[index].begin(), EventLists[index].end(),
		  pERin, timeOrder());
  EventLists[index].insert(iter, pERin);

  if (check == sctNb) {
    notEmpty = 0;
    for (index = 0; index < sctNb; index++)
      if (EventLists[index].size())
	notEmpty++;

    while (notEmpty == sctNb) {
      tOld = (u64) - 1;
      for (index = 0; index < sctNb; index++) {
	searcher = EventLists[index].begin();
	t[index] = getTimeOfThisEVENT(*searcher);
	if (t[index] < tOld) {
	  tOld = t[index];
	  old = index;
	}
      }
      destroyer = EventLists[old].begin();
      LMFbuilder(pEncoHC, pEHC, pCRHC, pGDHC, pcCC, *destroyer, pCRRC,
		 &pfC, fileNameC);

      freeER(*destroyer);
      EventLists[old].erase(destroyer);

      notEmpty = 0;
      for (index = 0; index < sctNb; index++)
	if (EventLists[index].size())
	  notEmpty++;
    }
  }
}

void finalCleanListEv()
{
  vector < EVENT_RECORD * >::iterator searcher, destroyer;

  u16 index, old = (u16) - 1;
  u8 notEmpty;
  u64 tOld;

  notEmpty = sctNb;
  while (notEmpty) {
    notEmpty = 0;
    tOld = (u64) - 1;
    for (index = 0; index < sctNb; index++)
      if (EventLists[index].size()) {
	searcher = EventLists[index].begin();

	t[index] = (u64) getTimeOfThisEVENT(*searcher);
	if (t[index] < tOld) {
	  tOld = t[index];
	  old = index;
	}
	notEmpty++;
      }

    if (notEmpty) {
      destroyer = EventLists[old].begin();
      LMFbuilder(pEncoHC, pEHC, pCRHC, pGDHC, pcCC,
		 *destroyer, pCRRC, &pfC, fileNameC);

      freeER(*destroyer);
      EventLists[old].erase(destroyer);
    }
  }

  delete[]EventLists;
  delete[]sctName;
  delete[]t;

  if (pEncoHC->scanContent.eventRecordBool == 1) {
    delete pEHC;
    if (pEncoHC->scanContent.gateDigiRecordBool == 1)
      if (pGDHC)
	delete pGDHC;
  }

  if (pcCC)
    delete pcCC;
  if (pEncoHC)
    delete pEncoHC;
  if (pCRHC)
    delete pCRHC;
  if (pCRRC)
    delete pCRRC;

  if (pfC)
    fclose(pfC);
}

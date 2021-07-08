/*-------------------------------------------------------

List Mode Format 
                        
--  multipleCoincidencesSorter.cc  --                      

Martin.Rey@epfl.ch
Crystal Clear Collaboration
Copyright (C) 2005 LPHE/EPFL, CH-1015 Lausanne

This software is distributed under the terms 
of the GNU Lesser General 
Public Licence (LGPL)
See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

Description of multipleCoincidencesSorter.cc:

sort all the coincidences (also the multiple ones)

-------------------------------------------------------*/

#include "timeOrder.hh"

#include <vector>
#include <algorithm>

using namespace std;

static vector < EVENT_RECORD * >EventList;

static ENCODING_HEADER *pEncoHC = NULL;
static EVENT_HEADER *pEHC = NULL;
static GATE_DIGI_HEADER *pGDHC = NULL;
static CURRENT_CONTENT *pcCC = NULL;
static EVENT_RECORD *pERC = NULL;
static int nN;			/* number of neighs */
static u32 cnt;			/* number of coinci */

static u64 stack_cut_time = 25;
static u64 cw = 10;
static u16 maxDifSct = 2;
static u8 multi = 0;

void setMultipleCoincidencesSorterParams()
{
  char response[10];

  printf("Set the coincidence window in nonoseconds (Default %llu ns): ",
	 cw);
  if (fgets(response, sizeof(response), stdin))
    sscanf(response, "%llu", &cw);

  printf("Set the stack cut time in microseconds (Default %llu micros): ",
	 stack_cut_time);
  if (fgets(response, sizeof(response), stdin))
    sscanf(response, "%llu", &stack_cut_time);

  printf
      ("Set the sector order of rejection for coincidences (Ex.: if 2\n");
  printf
      ("coincidences are accepted if |rsectorID1 - rsectorID2}| > 1 (Default %hu): ",
       maxDifSct);
  if (fgets(response, sizeof(response), stdin))
    sscanf(response, "%hu", &maxDifSct);

  response[0] = (multi) ? 'y' : 'n';

  printf
      ("Do you want to store multiple coincidences (y or n, default %c): ",
       response[0]);
  if (fgets(response, sizeof(response), stdin))
    multi = (response[0] == 'y') ? 1 : 0;

  cout << "response = " << response << " -> multi = " << (int) multi <<
      endl;

}

void setMultipleCoincidencesSorterParamsDirectly(u64 inline_cw, 
						 u64 inline_stack_cut_time,
						 u16 inline_maxDifSct,
						 u8 inline_multi)
{
  cw = inline_cw;
  stack_cut_time = inline_stack_cut_time;
  maxDifSct = inline_maxDifSct;
  multi = inline_multi;

  return;
}

void initMultipleCoincidencesSorter(const ENCODING_HEADER * pEncoH,
				    const EVENT_HEADER * pEH,
				    const GATE_DIGI_HEADER * pGDH,
				    const CURRENT_CONTENT * pcC)
{
  pEncoHC = new ENCODING_HEADER;
  pEHC = new EVENT_HEADER;
  pGDHC = new GATE_DIGI_HEADER;
  pcCC = new CURRENT_CONTENT;

  cnt = 0;

  if ((!pEncoHC) || (!pEHC) || (!pGDHC) || (!pcCC))
    cout <<
	"\n *** Error: multipleCoincidencesSorter.c: initMultipleCoincidencesSorter: new\n";

  if (pEncoH)
    *pEncoHC = *pEncoH;

  if (pEH) {
    *pEHC = *pEH;
    pEHC->coincidenceBool = 1;
    nN = pEH->numberOfNeighbours;
  }

  if (pGDH) {
    *pGDHC = *pGDH;		/*  no pointer in this structure, it is safe */
    pGDHC->multipleIDBool = 1;
  }

  if (pcC)
    *pcCC = *pcC;

  cout << "cw = " << cw;
  cw = cw * 1000000 / getTimeStepFromCCH();
  cout << " ns -> " << cw << endl;
  cout << "stack_cut_time = " << stack_cut_time;
  stack_cut_time = stack_cut_time * 1000000000 / getTimeStepFromCCH();
  cout << " micros -> " << stack_cut_time << endl;
  cout << "Max Diff sectors = " << maxDifSct << endl;
  return;
}

void multipleCoincidencesSorter(const ENCODING_HEADER * pEncoH,
				const EVENT_HEADER * pEH,
				const COUNT_RATE_HEADER * pCRH,
				const GATE_DIGI_HEADER * pGDH,
				const CURRENT_CONTENT * pcC,
				const EVENT_RECORD * pER,
				const COUNT_RATE_RECORD * pCRR)
{
  static u8 doneonce = 0;

  EVENT_RECORD *pERin;
  vector < EVENT_RECORD * >::iterator iter, first, second, last;

  u64 maxSize = 1000;
  u16 sct1, sct2, difSct = 0;
  u64 firstTime, lastTime;

  if (!doneonce) {
    initMultipleCoincidencesSorter(pEncoH, pEH, pGDH, pcC);
    pERC = newER(pEHC);

    doneonce++;
  }

  pERin = newER(pEH);
  copyER(pER, pERin, pEH);	/*   *pERin = *pER but safe */

  iter =
      lower_bound(EventList.begin(), EventList.end(), pERin, timeOrder());
  EventList.insert(iter, pERin);

  first = EventList.begin();
  last = EventList.end() - 1;

  firstTime = u8ToU64((*first)->timeStamp);
  lastTime = u8ToU64((*last)->timeStamp);

  /* if the conditions are respected
     (ie the list is bigger than maxSize and
     the time difference between the last and
     the first one is bigger than stack_cut_time)
     enter in the boucle and find coincidences */
  while ((EventList.size() > maxSize)
	 && (lastTime - firstTime > stack_cut_time)) {
    second = first + 1;

    if (second != EventList.end()) {
      sct1 = getRsectorID(pEncoHC, *first);
      sct2 = getRsectorID(pEncoHC, *second);
      difSct = (sct2 > sct1) ? sct2 - sct1 : sct1 - sct2;

      while ((u8ToU64((*second)->timeStamp) - firstTime <= cw)
	     && (difSct >= maxDifSct)) {
	//      pERC = fillCoinciRecord(first,second,multipleID);
	fillCoinciRecord(pEncoHC, pEHC, pGDHC, *first, *second, 0, nN, 0,
			 pERC);
	outputCoincidence(pEncoHC, pEHC, pGDHC, pcCC, pERC, 0);
	cnt++;

	second++;
	if (second != EventList.end())
	  break;
	sct2 = getRsectorID(pEncoHC, *second);
	difSct = (sct2 > sct1) ? sct2 - sct1 : sct1 - sct2;
      }
    }

    freeER(*first);
    EventList.erase(first);

    first = EventList.begin();
    firstTime = u8ToU64((*first)->timeStamp);
  }

  return;
}

void cleanMultipleCoincidencesSorterList()
{
  vector < EVENT_RECORD * >::iterator iter, first, second, last;
  u16 sct1, sct2, difSct = 0;
  u64 firstTime;

  cout << "List size = " << EventList.size() << endl;
  while (EventList.size()) {
    first = EventList.begin();
    second = first + 1;

    if (second != EventList.end()) {
      sct1 = getRsectorID(pEncoHC, *first);
      sct2 = getRsectorID(pEncoHC, *second);
      difSct = (sct2 > sct1) ? sct2 - sct1 : sct1 - sct2;

      firstTime = u8ToU64((*first)->timeStamp);
      while ((u8ToU64((*second)->timeStamp) - firstTime <= cw)
	     && (difSct >= maxDifSct)) {
	//      pERC = fillCoinciRecord(first,second,multipleID);
	fillCoinciRecord(pEncoHC, pEHC, pGDHC, *first, *second, 0, nN, 0,
			 pERC);
	outputCoincidence(pEncoHC, pEHC, pGDHC, pcCC, pERC, 0);
	cnt++;


	//      writeCoinci(*first, *second);
	//      EventList.erase(second);

	second++;
	if (second != EventList.end())
	  break;

	sct2 = getRsectorID(pEncoHC, *second);
	difSct = (sct2 > sct1) ? sct2 - sct1 : sct1 - sct2;
      }
    }

    freeER(*first);
    EventList.erase(first);
  }

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

  cout << cnt << " coincidences were written" << endl;
  cnt = 0;
}

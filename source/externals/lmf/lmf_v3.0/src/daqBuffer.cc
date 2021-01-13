/*-------------------------------------------------------

List Mode Format 
                        
--  daqBuffer.cc  --                      

Martin.Rey@epfl.ch

Crystal Clear Collaboration
Copyright (C) 2005 LPHE/EPFL, CH-1015 Lausanne

This software is distributed under the terms 
of the GNU Lesser General 
Public Licence (LGPL)
See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

/*-------------------------------------------------------

Description of daqBuffer (version c++) 

To mimics the effect of limited transfert rate, this module allows to
simulate the data loss due to an overflow of a memory buffer, read
periodically, following a given reading frequency. This module uses
three parameters, the reading frequency "nu", the buffer size "bf" and
the mode it works. Moreover, two reading methods can be modelized,
that is, in an event per event basis (an event is read at each reading
clock tick), or in a full buffer reading basic (at each reading clock
tick, the whole buffer is emptied out). In the first reading method,
the limit data rate is then limited to "nu", while in the second
method, the limit data rate is equal to "bf dot nu". When the size
limit is reached, any new pulse is rejected, until the next reading
clock tick arrival which will free a part of the buffer. In such a
case, a non null buffer depth allows to face to a local rise of the
input data flow.

-------------------------------------------------------*/

#include <stdio.h>
#include "lmf.h"

#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

/* class BufferPos used in daqBuffer function */

class BufferPos
{
public:
  BufferPos(u16 bufferNb, u16 sctByBufNb);
  ~BufferPos();
  void setSctIdx(vector<u16> sctIdx);
  u16 GetIdx(const u16 & sct) const;
  u32 operator()(const u16 & sct) const;
  u32 &operator()(const u16 & sct);

private:
  u16 m_bufferNb;
  u16 m_sctByBufNb;
  u32* m_bufferPos;
  u16** m_sct_values;
};

BufferPos::BufferPos(u16 bufferNb, u16 sctByBufNb = 2)
  : m_bufferNb(bufferNb), m_sctByBufNb(sctByBufNb)
{
  size_t i;

  m_bufferPos = new u32[bufferNb];
  for(i = 0; i < m_bufferNb; i++)
    m_bufferPos[i] = 0;
  m_sct_values = NULL;
}

BufferPos::~BufferPos()
{
  size_t i;

  for(i = 0; i < m_bufferNb; i++)
    cout << (int)i << "\t:" 
	 << (int)m_bufferPos[i] << endl;
  delete[] m_bufferPos;

  for(i = 0; i < m_sctByBufNb; i++)
    if(m_sct_values[i])
      delete[] m_sct_values[i];
  if(m_sct_values)
    delete[] m_sct_values;
}

void BufferPos::setSctIdx(vector<u16> sctIdx)
{
  size_t i, j;

  if(m_sct_values) {
    cout << "Already set -> exit" << endl;
    exit(EXIT_FAILURE);
  }

  if(sctIdx.size() != (size_t)(m_sctByBufNb * m_bufferNb)) {
    cout << "Wrong sct idx nb -> exit" << endl;
    exit(EXIT_FAILURE);
  }
  
  m_sct_values = new u16*[m_bufferNb];
  for(i = 0; i < m_bufferNb; i++) {
    m_sct_values[i] = new u16[m_sctByBufNb];
    for(j = 0; j < m_sctByBufNb; j++)
      m_sct_values[i][j] = sctIdx[i*m_sctByBufNb+j];
  }

  return;
}

u16 BufferPos::GetIdx(const u16 & sct) const
{
  size_t i, j;
  for(i = 0; i < m_bufferNb; i++)
    for(j = 0; j < m_sctByBufNb; j++)
      if( sct == m_sct_values[i][j])
	goto end;
 end:
  
  if(i == m_bufferNb) {
    cout << sct << " not in the sct idx list -> exit" << endl;
    exit(EXIT_FAILURE);
  }

  return i;
}

u32 BufferPos::operator()(const u16 & sct) const
{
  u16 idx;

  idx = GetIdx(sct);
  return m_bufferPos[idx];
}

u32 &BufferPos::operator()(const u16 & sct)
{
  u16 idx;

  idx = GetIdx(sct);
  return m_bufferPos[idx];
}

/* class BufferPos end */

static u8 mode = 0;
static u32 bufferSize = (u32)(-1);
static double read_freq = -1;
static BufferPos *bufferPos = NULL;
static u32 *old_clock = NULL;

static u16 bufferNb = 0;
static u32 tot = 0;
static u32 rej = 0;


void initDaqBuffer(u16 NiCards, u16 sctnb, u16 sct[])
{
  vector<u16> sctIdx;
  u16 i;

  bufferNb = NiCards;
  bufferPos = new BufferPos(bufferNb,(sctnb+1)/bufferNb);
  for(i=0;i<sctnb;i++)
    sctIdx.push_back(sct[i]);
  bufferPos->setSctIdx(sctIdx);

  return;
}


void setDaqBufferMode(u8 inlineMode)
{
  mode = inlineMode;

  return;
}

void setDaqBufferSize(u32 inlineBufferSize)
{
  bufferSize = inlineBufferSize;

  return;
}

void setDaqBufferReadingFrequency(double inlineFreq)
{
  read_freq = inlineFreq;

  return;
}

void daqBuffer(const ENCODING_HEADER *pEncoH,
	       EVENT_RECORD **ppER)
{
  vector<u16> sctIdx;
  u16 sct, idx, tmp;
  u16 sctNb = 0;
  static u64 time_begin = 0;
  u64 act_time;

  u32 clock, delta;

  if(!old_clock) {
    while(!bufferNb) {
      cout << "Set the reading buffer mode (0 or 1): " << endl;
      cin >> mode;
      cout << "Set the number of buffer in use: " << endl;
      cin >> bufferNb;
      while(!sctNb) {
	cout << "Set the number of sct in use: " << endl;
	cin >> sctNb;
      }
      bufferPos = new BufferPos(bufferNb,(sctNb+1)/bufferNb);


      for(sct = 0; sct < sctNb; sct++) {
	cout << "Set sct idx for buffer " << idx << ": " << endl;
	cin >> tmp;
	sctIdx.push_back(tmp);
      }
      bufferPos->setSctIdx(sctIdx);
    }
    old_clock = new u32[bufferNb];
    for(idx = 0; idx < bufferNb; idx++)
      old_clock[idx] = 0;

    if(bufferSize == (u32)(-1)) {
      cout << "Set the buffer Size: " << endl;
      cin >> bufferSize;
    }

    while(read_freq <= 0) {
      cout << "Set the reading frequency (in Hz): " << endl;
      cin >> read_freq;
    }

    cout << "Buffer size = " << bufferSize << endl;
    cout << "Reading Frequ = " << read_freq << " Hz -> ";
    read_freq = read_freq * ((double)(getTimeStepFromCCH()) /  1.E15);
    cout << read_freq << endl;

    time_begin = u8ToU64((*ppER)->timeStamp);
  }
  sct = getRsectorID(pEncoH, *ppER);
  idx = bufferPos->GetIdx(sct);

  act_time = u8ToU64((*ppER)->timeStamp);
  clock = (u32)(((double)(act_time - time_begin)) * read_freq);
  delta = (clock > old_clock[idx]) ? clock - old_clock[idx] : 0; 

  //   if(!idx) {
  //     cout << "act_time = " << act_time << endl;
  //     cout << "clock = " << clock << endl;
  //     cout << "old_clock = " << old_clock[idx] << endl;
  //     cout << "delta = " << delta << endl;
  //   }


  //   cout << (*bufferPos)(sct) << " -> ";
  switch(mode) {
  case 0: (*bufferPos)(sct) = ((*bufferPos)(sct) > delta) ? (*bufferPos)(sct) - delta : 0;
    break;
  case 1:
    if(delta > 0)
      (*bufferPos)(sct) = 0;
    break;
  }
  //   if(!sct) {
  //     cout << "bufferPos = " << (*bufferPos)(sct) << endl;
  //   }


  if((*bufferPos)(sct) < bufferSize)
    (*bufferPos)(sct)++;
  else {
    *ppER = NULL;
    rej++;
    //     if(!sct) {
    //       cout << "******************-> rejected" << endl;
    //       getchar();
    //     }
  }
  //   if(!sct) {
  //     cout << "-> " << (*bufferPos)(sct) << endl;
  //     getchar();
  //   }
  tot++;
  old_clock[idx] = clock;
  //   cout << (*bufferPos)(sct) << endl;
  //   getchar();
  return;
}

void daqBufferDestructor()
{
  //   u16 i;


  cout << endl << rej << " rejected on " << tot << " -> " << ((double)(rej))/tot * 100 << " %" << endl;


  delete bufferPos;
  delete[] old_clock;

  bufferSize = (u32)(-1);
  read_freq = -1;
  bufferPos = NULL;
  
  return;
}

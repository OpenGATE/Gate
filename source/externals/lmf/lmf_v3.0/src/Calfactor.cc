/*-------------------------------------------------------

List Mode Format 
                        
--  Calfactor.cc  --                      

Martin.Rey@epfl.ch

Crystal Clear Collaboration
Copyright (C) 2005 LPHE/EPFL, CH-1015 Lausanne

This software is distributed under the terms 
of the GNU Lesser General 
Public Licence (LGPL)
See LMF/LICENSE.txt for further details

-------------------------------------------------------*/

#include <fstream>
#include <iostream>
#include <stdio.h>
#include "lmf.h"
#include "Calfactor.hh"
#include "Gaussian.hh"

using namespace::std;

Calfactor::Calfactor(u8 itsNbSct,
		     u8 itsNbMod,
		     u8 itsNbCry,
		     u8 itsNbLay)
  : m_nbsct(itsNbSct),
    m_nbmod(itsNbMod),
    m_nbcry(itsNbCry),
    m_nblay(itsNbLay)
{
  u8 i, j, k;

  m_calfactors = new double*** [m_nbsct];
  for(i = 0; i < m_nbsct; i++) {
    m_calfactors[i] = new double** [m_nbmod];
    for(j = 0; j < m_nbmod; j++) {
      m_calfactors[i][j] = new double* [m_nbcry];
      for(k = 0; k < m_nbcry; k++)
	m_calfactors[i][j][k] = new double [m_nblay];
    }
  }

  m_calBase = new mean_std [m_nblay];
}

Calfactor::~Calfactor()
{
  u8 i, j, k;

  for(i = 0; i < m_nbsct; i++) {
    for(j = 0; j < m_nbmod; j++) {
      for(k = 0; k < m_nbcry; k++)
	delete []m_calfactors[i][j][k];
      delete []m_calfactors[i][j];
    }
    delete []m_calfactors[i];
  }
  delete []m_calfactors;
  delete[] m_calBase;
}

double Calfactor::operator()(const u8 & i, const u8 & j, const u8 & k, const u8 & l) const
{
//   u8 idx;

//   idx = SectorToIndex(i);
  return m_calfactors[i][j][k][l];
}

double &Calfactor::operator()(const u8 & i, const u8 & j, const u8 & k, const u8 & l)
{
//   u8 idx;

//   idx = SectorToIndex(i);
  return m_calfactors[i][j][k][l];
}

void Calfactor::ReadCalfactorTable(u8 sct, u8 mod, u8 lay, u8 set)
{
  ifstream in;
  char fileName[256];

  u8 cry;
  int tmp;
  double calfactor;

  sprintf(fileName,"%02d-%1d-%1d-%02d.cal", sct, mod, lay, set);
  in.open(fileName,ios::in);

  if (!(in.is_open())) {
    printf("Warning: file %s does not exist -> 0 set for all calib factor sct %hu mod %hu lay %hu !\n",fileName, sct, mod, lay);
    for (cry = 0; cry < m_nbcry; cry++)
      m_calfactors[sct][mod][cry][lay] = 0;
  }
  else {
    for (cry = 0; cry < m_nbcry; cry++) {
      in >> tmp;
      in >> calfactor;
      
      m_calfactors[sct][mod][cry][lay] = ENERGY_REF / calfactor;
    }
  
    in.close();
  }
  
  return;
}

void Calfactor::ReadAllCalfactorTables(u8 set)
{
  u8 sct, mod, lay;

  for(sct = 0; sct < m_nbsct; sct++)
    for(mod = 0; mod < m_nbmod; mod++)
      for(lay = 0; lay < m_nblay; lay++)
	ReadCalfactorTable(sct,mod,lay,set);
}

void Calfactor::WriteAllCalfactorInFiles(u8 set)
{
  ofstream out;
  char fileName[256];
  u8 sct, mod, cry, lay;

  for(sct = 0; sct < m_nbsct; sct++)
    for(mod = 0; mod < m_nbmod; mod++)
      for(lay = 0; lay < m_nblay; lay++) {
	sprintf(fileName,"%02d-%1d-%1d-%02d.cal", sct, mod, lay, set);
	out.open(fileName,ios::out);
	out.setf(ios_base::fixed, ios_base::floatfield);
	out.precision(0);

	for (cry = 0; cry < m_nbcry; cry++)
	  out << (int)cry << "\t" 
	      << m_calfactors[sct][mod][cry][lay] * ENERGY_REF << endl;
	out.close();
      }

  return;
}

u8 Calfactor::SectorToIndex(const u8 &sct) const
{
  u8 idx = 255;

  switch(sct) {
  case 0:
    idx = 0;
    break;
  case 1:
    idx = 1;
    break;
  case 7:
    idx = 2;
    break;
  case 8:
    idx = 3;
    break;
  case 9:
    idx = 4;
    break;
  case 10:
    idx = 5;
    break;
  }

  return idx;
}


u8 Calfactor::IndexToSector(const u8 &idx) const
{
  u8 sector = 255;

  switch(idx) {
  case 0:
    sector = 0;
    break;
  case 1:
    sector = 1;
    break;
  case 2:
    sector = 7;
    break;
  case 3:
    sector = 8;
    break;
  case 4:
    sector = 9;
    break;
  case 5:
    sector = 10;
    break;
  }

  return sector;
}

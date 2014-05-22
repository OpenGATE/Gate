/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


/*!
  \class  GateMuTable
  \author fabien.baldacci@creatis.insa-lyon.fr
 */

#ifndef GATEMUTABLES_HH
#define GATEMUTABLES_HH

#include "G4UnitsTable.hh"

class GateMuTable
{
public:
  GateMuTable(G4String name, G4int size);
  ~GateMuTable();
  void PutValue(int index, double energy, double mu, double mu_en);
  double GetMuEn(double energy);
  double GetMu(double energy);
  G4int GetSize();
  double* GetEnergies() {return mEnergy;}
  double* GetMuEnTable() {return mMu_en;}
  double* GetMuTable() {return mMu;}

private:
  double *mEnergy;
  double *mMu;
  double *mMu_en;
  double lastEnergyMu;
  double lastEnergyMuen;
  double lastMu;
  double lastMuen;
  G4int mSize;
};


#endif

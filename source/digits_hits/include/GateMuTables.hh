/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


/*!
  \class  GateMuTable
  \author fabien.baldacci@creatis.insa-lyon.fr
 */

#ifndef GATEMUTABLES_HH
#define GATEMUTABLES_HH

#include "G4UnitsTable.hh"
#include "G4MaterialCutsCouple.hh"
#include "G4Material.hh"

class GateMuTable
{
public:
  GateMuTable(const G4MaterialCutsCouple *couple, G4int size);
  ~GateMuTable();
  void PutValue(int index, double energy, double mu, double mu_en);
  
  double GetMuEn(double energy);
  double GetMuEnOverRho(double energy);
  double GetMu(double energy);
  double GetMuOverRho(double energy);
  
  const G4MaterialCutsCouple *GetMaterialCutsCouple() { return mCouple; }
  const G4Material *GetMaterial() { return mMaterial; }
  double GetDensity() { return mDensity; }
  
  G4int GetSize() { return mSize; }
  double* GetEnergies() {return mEnergy;}
  double* GetMuEnTable() {return mMu_en;}
  double* GetMuTable() {return mMu;}

private:
  
  const G4MaterialCutsCouple *mCouple;
  const G4Material *mMaterial;
  double mDensity;
  
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

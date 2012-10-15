#ifndef GATEMUTABLES_CC
#define GATEMUTABLES_CC

#include "GateMuTables.hh"
#include "GateMiscFunctions.hh"
GateMuTable::GateMuTable(G4String /*name*/, G4int size)
{
  mEnergy = new double[size];
  mMu = new double[size];
  mMu_en = new double[size];
  mSize = size;
  lastMuen = -1.0;
  lastEnergy = -1.0;
}

GateMuTable::~GateMuTable()
{
  delete[] mEnergy;
  delete[] mMu;
  delete[] mMu_en;
}

void GateMuTable::PutValue(int index, double energy, double mu, double mu_en)
{
  mEnergy[index] = energy;
  mMu[index] = mu;
  mMu_en[index] = mu_en;
}

inline double interpol(double e_low, double e, double e_high,
		       double mu_en_low, double mu_en_high)
{
  return exp( log(mu_en_low) + log(mu_en_high/mu_en_low) / log(e_high/e_low)* log(e/e_low) );
}

double GateMuTable::GetMuEn(double energy)
{
  if (energy == lastEnergy)
    return lastMuen;
  lastEnergy = energy;
  int inf = 0, up = mSize-1;
  while(up - inf > 1){
    int tmp_bound = (inf + up)/2;
    if(mEnergy[tmp_bound] > energy)
      up = tmp_bound;
    else
      inf = tmp_bound;
  }
  double e_low = mEnergy[inf], e_high = mEnergy[up];
  if( energy > e_low && energy < e_high)
    lastMuen = interpol(e_low, energy, e_high, mMu_en[inf], mMu_en[up]);
  else
    lastMuen = mMu_en[inf];
  
  return lastMuen;
}

double GateMuTable::GetMu(double energy)
{
  int inf = 0, up = mSize-1;
  while(up - inf > 1){
    int tmp_bound = (inf + up)/2;
    if(mEnergy[tmp_bound] > energy)
      up = tmp_bound;
    else
      inf = tmp_bound;
  }
  double e_low = mEnergy[inf], e_high = mEnergy[up];
  if( energy > e_low && energy < e_high)
    return interpol(e_low, energy, e_high, mMu[inf], mMu[up]);
  else
    return mMu[inf];
}


G4int GateMuTable::GetSize()
{
  return mSize;
}

#endif

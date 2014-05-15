#ifndef GATEMUTABLES_CC
#include "GateMuTables.hh"
#include "GateMiscFunctions.hh"
GateMuTable::GateMuTable(G4String /*name*/, G4int size)
{
  mEnergy = new double[size];
  mMu = new double[size];
  mMu_en = new double[size];
  mSize = size;
  lastMu = -1.0;
  lastMuen = -1.0;
  lastEnergyMu = -1.0;
  lastEnergyMuen = -1.0;
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

inline double interpol(double x1, double x, double x2,
		       double y1, double y2)
{
  return ( y1 + ( (y2-y1) * (x-x1) / (x2-x1) ) );  // if log storage
//   return exp( log(y1) + log(y2/y1) / log(x2/x1)* log(x/x1) ); // if no log storage
}

double GateMuTable::GetMuEn(double energy)
{
  if (energy != lastEnergyMuen)
  {
    lastEnergyMuen = energy;

    energy = log(energy);

    int inf = 0;
    int sup = mSize-1;
    while(sup - inf > 1)
    {
      int tmp_bound = (inf + sup)/2;
      if(mEnergy[tmp_bound] > energy) { sup = tmp_bound; }
      else { inf = tmp_bound; }
    }
    double e_inf = mEnergy[inf];
    double e_sup = mEnergy[sup];

    if( energy > e_inf && energy < e_sup) { lastMuen = exp(interpol(e_inf, energy, e_sup, mMu_en[inf], mMu_en[sup])); }
    else { lastMuen = exp(mMu_en[inf]); }    
  }

  return lastMuen;
}

double GateMuTable::GetMu(double energy)
{
  if (energy != lastEnergyMu)
  {
    lastEnergyMu = energy;

    energy = log(energy);

    int inf = 0;
    int sup = mSize-1;
    while(sup - inf > 1)
    {
      int tmp_bound = (inf + sup)/2;
      if(mEnergy[tmp_bound] > energy) { sup = tmp_bound; }
      else { inf = tmp_bound; }
    }
    double e_inf = mEnergy[inf];
    double e_sup = mEnergy[sup];

    if( energy > e_inf && energy < e_sup) { lastMu = exp(interpol(e_inf, energy, e_sup, mMu[inf], mMu[sup])); }
    else { lastMu = exp(mMu[inf]); }
  }
    
  return lastMu;
}

G4int GateMuTable::GetSize()
{
  return mSize;
}

#endif

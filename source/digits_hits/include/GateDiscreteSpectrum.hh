/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "GateConfiguration.h"
#ifdef G4ANALYSIS_USE_ROOT

/*!
  \class  GateDiscreteSpectrumActor. Store a list of discrete energies and an associated quantity.
*/

#ifndef GATEDISCRETESPECTRUM_HH
#define GATEDISCRETESPECTRUM_HH

#include "GateMessageManager.hh"
#include <vector>
#include <algorithm>

class GateDiscreteSpectrum {
public:

  GateDiscreteSpectrum() { SetTolerance(0.0001); } /// 0.1 eV
  void Fill(double energy, double weight);
  void FillBin(int bin, double w);
  int GetDiscreteEnergy(double e) const;
  int AddDiscreteEnergy(double e);
  double GetValueAtEnergy(double e) const { return spectrum[GetDiscreteEnergy(e)].second; } // Warning, o not check bounds !
  double GetValue(int bin) const { return spectrum[bin].second; }
  double GetEnergy(int bin) const { return spectrum[bin].first; }
  int size() const { return spectrum.size(); }
  void SetTolerance(double d) { tolerance = d; }

  typedef std::pair<double, double> EnergyValuePair;

protected:
  std::vector<EnergyValuePair> spectrum;
  double tolerance;

  static bool cmp(const EnergyValuePair & a1, const double & a2) {
    return (a1.first < a2);
  }
};




#endif /* end #define GATEDISCRETESPECTRUM_HH */
#endif

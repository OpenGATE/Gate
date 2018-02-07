/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "GateDiscreteSpectrum.hh"


//-----------------------------------------------------------------------------
void GateDiscreteSpectrum::Fill(double e, double w)
{
  int bin = GetDiscreteEnergy(e);
  if (bin != -1) FillBin(bin, w);
  else {
    bin = AddDiscreteEnergy(e);
    FillBin(bin, w);
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
int GateDiscreteSpectrum::GetDiscreteEnergy(double e) const
{
  if (spectrum.size() == 0) return -1;
  double max_diff = tolerance;
  double diff;
  std::vector<EnergyValuePair>::const_iterator low;
  low = std::lower_bound(spectrum.begin(), spectrum.end(), e, GateDiscreteSpectrum::cmp);
  if (low == spectrum.end()) { // e is larger than current max E
    diff = e-spectrum.back().first;
    if (diff > max_diff) return -1;
    else return spectrum.size()-1; // slightly larger but inside last bin
  }
  diff = low->first-e;
  if (diff < max_diff) return low-spectrum.begin(); // inside that bin
  if (low != spectrum.begin()) { // check previous bin
    --low;
    diff = e-low->first;
    if (diff < max_diff) return low-spectrum.begin();
  }
  return -1;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateDiscreteSpectrum::FillBin(int bin, double w) {
  spectrum[bin].second += w;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
int GateDiscreteSpectrum::AddDiscreteEnergy(double e)
{
  spectrum.push_back(std::make_pair(e, 0.0));
  std::sort(spectrum.begin(), spectrum.end());
  return GetDiscreteEnergy(e);
}
//-----------------------------------------------------------------------------

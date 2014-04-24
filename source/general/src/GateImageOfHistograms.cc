/*----------------------
  GATE version name: gate_v7

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GateImageOfHistograms.hh"
#include "GateMiscFunctions.hh"

//-----------------------------------------------------------------------------
GateImageOfHistograms::GateImageOfHistograms():GateImage()
{
  DD("GateImageOfHistograms constructor");
  SetHistoInfo(0,0,0);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateImageOfHistograms::~GateImageOfHistograms()
{
  DD("GateImageOfHistograms:: destructor");
  mHistoData.clear();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateImageOfHistograms::SetHistoInfo(int n, double min, double max)
{
  gamma_bin = n;
  min_gamma_energy = min;
  max_gamma_energy = max;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateImageOfHistograms::Allocate()
{
  mHistoData.resize(nbOfValues);
  mHistoData.resize(nbOfValues);
  DD(nbOfValues);
  DD(gamma_bin);
  DD(min_gamma_energy);
  DD(max_gamma_energy);
  for(int i=0; i<nbOfValues; i++) {
    // Create TH1D with no names (to save memory)
    mHistoData[i] = new TH1D("","", gamma_bin, min_gamma_energy, max_gamma_energy);
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateImageOfHistograms::Reset()
{
  std::vector<TH1D*>::iterator iter = mHistoData.begin();
  while (iter != mHistoData.end()) {
    (*iter)->Reset();
    ++iter;
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateImageOfHistograms::AddValue(const int & index, TH1D * h)
{
  // DD(index);
  //DD(mHistoData[index]->GetNbinsX());
  //DD(h->GetNbinsX());
  //  DD(h->GetEntries());
  //DD(mHistoData[index]->GetEntries());
  mHistoData[index]->Add(h);
  //DD(mHistoData[index]->GetEntries());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateImageOfHistograms::Write(G4String filename, const G4String & comment)
{
  DD("GateImageOfHistograms::Write");
  DD(filename);

  G4String extension = getExtension(filename);
  DD(extension);

  // FIXME: convert histo into scalar image
  data.resize(nbOfValues);
  std::fill(data.begin(), data.end(), 0.0);
  unsigned long index = 0;
  for(unsigned int k=0; k<resolution.z(); k++) {
    for(unsigned int j=0; j<resolution.y(); j++) {
      for(unsigned int i=0; i<resolution.x(); i++) {
        data[index] = mHistoData[index]->GetEntries();
        // DD(index);
        // DD(mHistoData[index]->GetEntries());
        index++;
      }
    }
  } // end loop

  GateImage::Write(filename, comment);

}
//-----------------------------------------------------------------------------

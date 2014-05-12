/*----------------------
  GATE version name: gate_v7

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GateImageOfHistograms.hh"
#include "GateMiscFunctions.hh"
#include <TFile.h>

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

  // Could not allocate this way for 3D image -> too long !!
  /*
  for(int i=0; i<nbOfValues; i++) {
    // Create TH1D with no names (to save memory)
    //DD(i);
    mHistoData[i] = new TH1D("","", gamma_bin, min_gamma_energy, max_gamma_energy);
  }
  */

  // mHistoData = new

  // DEBUG
  mTotalEnergySpectrum = new TH1D("","", gamma_bin, min_gamma_energy, max_gamma_energy);


  DD("end hist created");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateImageOfHistograms::Reset()
{
  DD("Reset");
  std::vector<TH1D*>::iterator iter = mHistoData.begin();
  while (iter != mHistoData.end()) {
    // Check if allocated because on the fly allocation
    if (*iter) (*iter)->Reset();
    ++iter;
  }
  DD("end Reset");
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

  // Dynamic allocation
  // DD(index);

  // On the fly allocation
  if (!mHistoData[index]) {
    // DD(index);
    mHistoData[index] = new TH1D("","", gamma_bin, min_gamma_energy, max_gamma_energy);
  }

  // The overhead of a TH1 is about 600 bytes + the bin contents,

  mHistoData[index]->Add(h);
  //DD(mHistoData[index]->GetEntries());

  // TOTAL H FIXME
  mTotalEnergySpectrum->Add(h);

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
  unsigned long nb_non_null = 0;
  for(unsigned int k=0; k<resolution.z(); k++) {
    for(unsigned int j=0; j<resolution.y(); j++) {
      for(unsigned int i=0; i<resolution.x(); i++) {
        //        data[index] = mHistoData[index]->GetEntries();
        if (mHistoData[index]) { // because on the fly allocation
          data[index] = mHistoData[index]->GetSumOfWeights();// same getsum but exclude under/overflow
          //data[index] = mHistoData[index]->GetSum();//Entries();
          //data[index] = mHistoData[index]->GetEntries();
          nb_non_null++;
        }
        //else data[index] = 0.0; // no need because filled
        // DD(index);
        // DD(mHistoData[index]->GetEntries());
        index++;
      }
    }
  } // end loop
  DD(nb_non_null);

  GateImage::Write(filename, comment);

  DD("write root");
  TFile * pTfile = new TFile("total.root","RECREATE");
  mTotalEnergySpectrum->Write();
  DD("write root end");
}
//-----------------------------------------------------------------------------

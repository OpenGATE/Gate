/*----------------------
  GATE version name: gate_v7

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

// Gate
#include "GateImageOfHistograms.hh"
#include "GateMiscFunctions.hh"

// Root
#include <TFile.h>

// From ITK
#include "metaObject.h"
#include "metaImage.h"


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
  //mHistoData.clear();
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
  // mHistoData.resize(nbOfValues);
  // mHistoData.resize(nbOfValues);
  DD(nbOfValues);
  DD(gamma_bin);
  DD(min_gamma_energy);
  DD(max_gamma_energy);


  //  FIXME : display memory size

  // Could not allocate this way for 3D image -> too long !!
  /*
  for(int i=0; i<nbOfValues; i++) {
    // Create TH1D with no names (to save memory)
    //DD(i);
    mHistoData[i] = new TH1D("","", gamma_bin, min_gamma_energy, max_gamma_energy);
  }
  */

  // mHistoData = new
  dataDouble.resize(nbOfValues * gamma_bin);
  DD(dataDouble.size());
  Reset(); //fill(data.begin(), data.end(), 0.0);

  // DEBUG
  mTotalEnergySpectrum = new TH1D("","", gamma_bin, min_gamma_energy, max_gamma_energy);


  DD("end hist created");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateImageOfHistograms::Reset()
{
  DD("Reset");
  // std::vector<TH1D*>::iterator iter = mHistoData.begin();
  // while (iter != mHistoData.end()) {
  //   // Check if allocated because on the fly allocation
  //   if (*iter) (*iter)->Reset();
  //   ++iter;
  // }
  fill(dataDouble.begin(), dataDouble.end(), 0.0);
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

  /*
  // On the fly allocation
  if (!mHistoData[index]) {
    // DD(index);
    mHistoData[index] = new TH1D("","", gamma_bin, min_gamma_energy, max_gamma_energy);
  }

  // The overhead of a TH1 is about 600 bytes + the bin contents,
  mHistoData[index]->Add(h);
  //DD(mHistoData[index]->GetEntries());
  */

  int index_data = index*gamma_bin;
  for(unsigned int i=1; i<=gamma_bin; i++) {
    dataDouble[index_data] += h->GetBinContent(i); // +1 because TH1D start at 1, and end at index=size
    index_data++;
  }

  // TOTAL H FIXME
  mTotalEnergySpectrum->Add(h);

}
//-----------------------------------------------------------------------------


#define DDV(a,n) { GateMessage("Core", 0 , #a " = [ "; for(unsigned int _i_=0; _i_<n; _i_++) { std::cout << a[_i_] << " "; }; std::cout << " ]" << Gateendl);}

//-----------------------------------------------------------------------------
void GateImageOfHistograms::Write(G4String filename, const G4String & comment)
{
  DD("GateImageOfHistograms::Write");
  DD(filename);

  std::string extension = getExtension(filename);
  std::string baseName = removeExtension(filename);
  DD(extension);
  DD(baseName);

  // FIXME : FIRST version with FULL raw data ; next to change in
  // dynamic allocation, keep empty check extension type mhd ==> no
  // change to ioh (ImageOfHistogram) ?

  // Create a mhd header : 4D image with the 1th dimension is the gamma Energy spectrum
  int * dimSize = new int[4];
  float * spacing = new float[4];
  dimSize[0] = GetResolution().x();
  dimSize[1] = GetResolution().y();
  dimSize[2] = GetResolution().z();
  dimSize[3] = gamma_bin; // Gamma energy spectrum bins
  spacing[0] = GetVoxelSize().x();
  spacing[1] = GetVoxelSize().x();
  spacing[2] = GetVoxelSize().x();
  spacing[3] = 1; // Dummy
  MetaImage m_MetaImage(4, dimSize, spacing, MET_DOUBLE);
  DDV(dimSize, 4);
  DDV(spacing, 4);

  std::string headerName = filename;
  std::string rawName = baseName+".raw";
  DD(filename);
  DD(headerName);
  DD(rawName);

  // Copy from GateMHDImage
  double p[3];
  // Gate convention: origin is the corner of the first pixel
  // MHD / ITK convention: origin is the center of the first pixel
  // -> Add a half pixel
  p[0] = GetOrigin().x() + GetVoxelSize().x()/2.0;
  p[1] = GetOrigin().y() + GetVoxelSize().y()/2.0;
  p[2] = GetOrigin().z() + GetVoxelSize().z()/2.0;
  m_MetaImage.Position(p);
  DDV(p, 3);

  // Transform
  double matrix[16];
  for(unsigned int i=0; i<3; i++) {
    matrix[i*4  ] = GetTransformMatrix().row1()[i];
    matrix[i*4+1] = GetTransformMatrix().row2()[i];
    matrix[i*4+2] = GetTransformMatrix().row3()[i];
    matrix[i*4+3] = 0.0;
  }
  matrix[12] = matrix[13] = matrix[14] = 0.0;
  matrix[15] = 1.0;
  DDV(matrix, 16);
  m_MetaImage.TransformMatrix(matrix);


  DD("copy data");
  std::vector<double> t;
  t.resize(nbOfValues*gamma_bin);
  std::fill(t.begin(), t.end(), 0.0);
  unsigned long index_image = 0;
  unsigned long index_data = 0;
  for(int l=0; l<gamma_bin; l++) {
    index_data = l;
    for(unsigned int k=0; k<resolution.z(); k++) {
      for(unsigned int j=0; j<resolution.y(); j++) {
        for(unsigned int i=0; i<resolution.x(); i++) {
          t[index_image] = dataDouble[index_data];
          index_image++;
          index_data +=gamma_bin;
        }
      }
    }
  }

  DD("Write data");
  DD(data.size()); // the size is X x Y x Z x gamma_bin
  m_MetaImage.ElementData(&(t.begin()[0]), false); // true = autofree
  m_MetaImage.Write(headerName.c_str(), rawName.c_str());

  ///-----------------------------------------
  // Below Additional debug output : will be removed
  ///-----------------------------------------

  // convert histo into scalar image
  std::vector<float> temp;
  temp.resize(nbOfValues);
  std::fill(temp.begin(), temp.end(), 0.0);
  index_image = 0;
  index_data = 0;
  for(unsigned int k=0; k<resolution.z(); k++) {
    for(unsigned int j=0; j<resolution.y(); j++) {
      for(unsigned int i=0; i<resolution.x(); i++) {
        for(int l=0; l<gamma_bin; l++) {
          temp[index_image] += dataDouble[index_data];
          index_data++;
        }
        index_image++;
      }
    }
  }
  headerName = baseName+"_sum.mhd";
  rawName    = baseName+"_sum.raw";
  DD(headerName);
  DD(rawName);
  MetaImage mm(3, dimSize, spacing, MET_FLOAT);
  mm.Position(p);
  double matrix2[9];
  for(unsigned int i=0; i<3; i++) {
    matrix2[i*3  ] = GetTransformMatrix().row1()[i];
    matrix2[i*3+1] = GetTransformMatrix().row2()[i];
    matrix2[i*3+2] = GetTransformMatrix().row3()[i];
  }
  DDV(matrix2, 9);
  mm.TransformMatrix(matrix2);
  mm.ElementData(&(temp.begin()[0]), false); // true = autofree
  mm.Write(headerName.c_str(), rawName.c_str());

  DD("write root");
  TFile * pTfile = new TFile("total.root","RECREATE");
  mTotalEnergySpectrum->Write();
  DD("write root end");
}
//-----------------------------------------------------------------------------

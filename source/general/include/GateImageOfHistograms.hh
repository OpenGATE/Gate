/*----------------------
  GATE version name: gate_v7

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GateConfiguration.h"

#ifndef GATEIMATEOFHISTOGRAMS_HH
#define GATEIMATEOFHISTOGRAMS_HH

#include "GateImage.hh"
#include <TH1D.h>

/*
  This class GateImageOfHistograms manages an image with a 1D
  histogram in each pixel. The histogram properties must be the same
  in each pixel (same min/max number of bins). The class inherit from
  GateImage (for size, spacing etc), but do not use the data member
  because it is float. Data are stored in dataDouble member as a raw
  vector. In memory, the data are ordered as H X Y Z, with H for
  histogram, in order to access all histogram value in a pixel in a
  consecutive way. However on disk (mhd file format), data are
  re-ordered according to X Y Z H like in conventional 4D image.

  WARNING : all function inherited from GateImage that access the
  initial 'data' member will result in a seg fault. Use
  GetDataDoublePointer.

  FIXME:
  - manage a sparse version (not an histo at every pixel)


 */

//-----------------------------------------------------------------------------
class GateImageOfHistograms:public GateImage
{
public:

  GateImageOfHistograms();
  ~GateImageOfHistograms();

  void SetHistoInfo(int n, double min, double max);
  virtual void Allocate();
  void Reset();
  void AddValue(const int & index, TH1D * h);
  virtual void Write(G4String filename, const G4String & comment = "");
  virtual void Read(G4String filename);
  unsigned int GetNbOfBins() { return nbOfBins; }
  double GetMaxValue() { return maxValue; }
  double GetMinValue() { return minValue; }
  double * GetDataDoublePointer() { return &dataDouble[0]; }
  long GetIndexFromPixelIndex(int i, int j, int k);
  virtual void UpdateSizesFromResolutionAndHalfSize();
  virtual void UpdateSizesFromResolutionAndVoxelSize();

  // Compute and image (data only) of the sum of histo by pixel (in order HXYZ)
  void ComputeTotalOfCountsImageData(std::vector<double> & output);

protected:
  double minValue;
  double maxValue;
  unsigned int nbOfBins;
  std::vector<double> dataDouble;

  // Store a copy of G4ThreeVector resolution in int for integer
  // computation of index
  long sizeX;
  long sizeY;
  long sizeZ;
  long sizePlane;

  // Data pixel order
  void ConvertPixelOrderToXYZH(std::vector<double> & input, std::vector<double> & output);
  void ConvertPixelOrderToHXYZ(std::vector<double> & input, std::vector<double> & output);

  // DEBUG
  std::vector<TH1D*> mHistoData; // FIXME : debug for sparse version
  TH1D * mTotalEnergySpectrum;
};
//-----------------------------------------------------------------------------

#endif // GATEIMAGEOFHISTOGRAMS

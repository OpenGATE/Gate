/*----------------------
  GATE version name: gate_v7

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
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

  GateImageOfHistograms(std::string dataTypeName);
  ~GateImageOfHistograms();

  void SetHistoInfo(int n, double min, double max);
  virtual void Allocate();
  void Reset();
  void AddValueFloat(const int & index, TH1D * h, const double scale);
  void AddValueDouble(const int & index, TH1D * h, const double scale);
  void AddValueDouble(const int & index, const int &bin, const double value);
  void AddValueDouble(const int & index, const double value, const double scale); /** Modif Oreste **/
  void SetValueDouble(const int & index, const int &bin, const double value);
  double GetValueDouble(const int & index, const int &bin);
  void AddValueInt(const int & index, const int &bin, const unsigned int value);
  virtual void Write(G4String filename, const G4String & comment = "");
  virtual void Read(G4String filename);
  unsigned int GetNbOfBins() { return nbOfBins; }
  double GetMaxValue() { return maxValue; }
  double GetMinValue() { return minValue; }
  unsigned long GetDoubleSize() { return dataDouble.size(); }
  unsigned long GetFloatSize() { return dataFloat.size(); }
  unsigned long GetIntSize() { return dataInt.size(); }
  double * GetDataDoublePointer() { return &dataDouble[0]; }
  float * GetDataFloatPointer() { return &dataFloat[0]; }
  unsigned int * GetDataIntPointer() { return &dataInt[0]; }
  long GetIndexFromPixelIndex(int i, int j, int k);
  virtual void UpdateSizesFromResolutionAndHalfSize();
  virtual void UpdateSizesFromResolutionAndVoxelSize();
  void Scale(double f);
  double ComputeSum();
  void Deallocate();

  // Compute and image (data only) of the sum of histo by pixel (in order HXYZ)
  void ComputeTotalOfCountsImageDataFloat(std::vector<float> & output);
  void ComputeTotalOfCountsImageDataDouble(std::vector<double> & output);

protected:
  double minValue;
  double maxValue;
  unsigned int nbOfBins;

  // Data can be stored in double or float. Data always write/read in
  // float.
  std::string mDataTypeName;
  std::vector<double> dataDouble;
  std::vector<float> dataFloat;
  std::vector<unsigned int> dataInt;

  // Store a copy of G4ThreeVector resolution in int for integer
  // computation of index
  long sizeX;
  long sizeY;
  long sizeZ;
  long sizePlane;

  // Data pixel order
  template<class PT>
  void ConvertPixelOrderToXYZH(std::vector<PT> & input, std::vector<PT> & output);
  template<class PT>
  void ConvertPixelOrderToHXYZ(std::vector<PT> & input, std::vector<PT> & output);

};
//-----------------------------------------------------------------------------

// For templated functions
#include "GateImageOfHistograms.icc"

#endif // GATEIMAGEOFHISTOGRAMS

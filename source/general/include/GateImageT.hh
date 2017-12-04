/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/


/*!
  \class GateImageT
  \ingroup data_structures
  \author thibault.frisson@creatis.insa-lyon.fr
  laurent.guigues@creatis.insa-lyon.fr
  david.sarrut@creatis.insa-lyon.fr
*/

#ifndef __GATEIMAGET_HH__
#define __GATEIMAGET_HH__

// g4
#include <globals.hh>

// std
#include <fstream>
#include <iomanip>

// gate
#include "GateVImage.hh"
#include "GateMachine.hh"
#include "GateMHDImage.hh"
#include "GateDICOMImage.hh"
#include "GateMiscFunctions.hh"

// root
#ifdef G4ANALYSIS_USE_ROOT
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#endif


/// \brief 3D images of PixelType values
template<class PixelType>
class GateImageT : public GateVImage
{
public:

  GateImageT();
  virtual ~GateImageT();

  // Define some iterator types
  typedef typename std::vector<PixelType>::iterator iterator;
  typedef typename std::vector<PixelType>::const_iterator const_iterator;

  /// Allocates the data
  virtual void Allocate();

  // Access to the image values
  /// Returns the value of the image at voxel of index provided
  inline PixelType GetValue(int index) const { return data[index]; }

  /// Returns a reference on the value of the image at voxel of index provided
  inline PixelType& GetValue(int index)       { return data[index]; }

  /// Returns the value of the image at voxel of coordinates provided
  inline PixelType GetValue(int i, int j, int k) const { return data[i+j*lineSize+k*planeSize]; }

  /// Returns the value of the image at voxel of position provided
  inline PixelType GetValue(const G4ThreeVector& position) const { return data[GetIndexFromPosition(position)]; }

  /// Returns a reference on the value of the image at voxel of position provided
  inline PixelType& GetValue(const G4ThreeVector& position) { return data[GetIndexFromPosition(position)]; }
  /// Sets the value of the voxel of coordinates x,y,z

  inline void SetValue ( int x, int y, int z, PixelType v ) { data[x+y*lineSize+z*planeSize]=v; }
  /// Sets the value of the voxel of index i

  inline void SetValue ( int i, PixelType v ) { data[i]=v; }

  /// Adds a value to the voxel of index provided
  inline void AddValue(int index, PixelType value) { data[index] += value; }

  /// Fills the image with a value
  //  inline void Fill(PixelType v) { for (iterator i=begin(); i!=end(); ++i) (*i)=v; }
  inline void Fill(PixelType v) { fill(data.begin(), data.end(), v); }

  inline PixelType GetMinValue() const{ return *std::min_element(begin(), end()); }
  inline PixelType GetMaxValue() const{ return *std::max_element(begin(), end()); }

  inline PixelType GetOutsideValue()   { return mOutsideValue; } //The HU value that must be considered not part of the phantom
  inline void SetOutsideValue( PixelType v ) { mOutsideValue=v; }
  PixelType GetNeighborValueFromCoordinate(const ESide & side, const G4ThreeVector & coord);

  void MergeDataByAddition(G4String filename);

  // iterators
  iterator begin() { return data.begin(); }
  iterator end()   { return data.end(); }
  const_iterator begin() const { return data.begin(); }
  const_iterator end() const  { return data.end(); }

  // IO
  /// Writes the image to a file with comment (the format is detected automatically)
  virtual void Write(G4String filename, const G4String & comment = "");

  /// Reads the image from a file (the format is detected automatically)
  virtual void Read(G4String filename);


  /// Displays info about the image to standard output
  virtual void PrintInfo();


  //-----------------------------------------------------------------------------
protected:
  std::vector<PixelType> data;
  PixelType mOutsideValue;

  void ReadAscii(G4String filename);
  void ReadAnalyze(G4String filename);
  void ReadMHD(G4String filename);
  void ReadInterfile(G4String fileName);
  void ReadDICOM(G4String fileName);

  void WriteVox(std::ofstream & os);
  void WriteAscii(std::ofstream & os, const G4String & comment);
  void WriteBin(std::ofstream & os);
  void WriteAnalyzeHeader(G4String filename);
  void WriteRoot(G4String filename);
  void WriteMHD(std::string filename);
  void WriteDICOM(std::string filename);

};

#include "GateImageT.icc"

#endif

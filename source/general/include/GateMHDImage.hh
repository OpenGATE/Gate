/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#ifndef __GATEMHDIMAGE_HH__
#define __GATEMHDIMAGE_HH__

// g4
#include "globals.hh"
#include "G4ThreeVector.hh"

// std
#include <vector>

// gate
#include "GateMessageManager.hh"

// itk (for mhd reader)
//#include "metaObject.h"
#include "metaImage.h"

template<class PixelType> class GateImageT;

//-----------------------------------------------------------------------------
// Read an write MHD image file format. Use the metaImageIO from ITK.

class GateMHDImage
{
public:

  GateMHDImage();
  ~GateMHDImage();

  void ReadHeader(std::string & filename);
  template<class PixelType>
  void ReadData(std::string filename, std::vector<PixelType> & data);

  template<class PixelType>
  void WriteHeader(std::string filename,
                   GateImageT<PixelType> * image,
                   bool writeData = false,
                   bool changeExtension = false,
                   bool isARF = false,
                   int numberOfARFFFDHeads = 1);
  template<class PixelType>
  void WriteData(std::string filename, GateImageT<PixelType> * image);

  std::vector<double> size;
  std::vector<double> spacing;
  std::vector<double> origin;
  std::vector<double> transform;
  //-----------------------------------------------------------------------------
protected:
  std::vector<std::string> tags;
  std::vector<std::string> values;

  void Print();
  void Read_3_values(std::string tag, double * v);
  int Read_tag(std::string tag);
  void Check_tag_value(std::string tag, std::string value);
  void EraseWhiteSpaces(std::string & s);
  void GetRawFilename(std::string filename,
                      std::string & f,
                      bool keepFolder,
                      bool changeExtension = false);
  double round_to_digits(double, int);

};

#include "GateMHDImage.icc"

#endif

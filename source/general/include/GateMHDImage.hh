/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#ifndef __GATEMHDIMAGE_HH__
#define __GATEMHDIMAGE_HH__

#include "globals.hh"
#include "G4ThreeVector.hh"
#include <vector>

#include "GateMessageManager.hh"
#include "GateImage.hh"

//-----------------------------------------------------------------------------
// Read an write MHD image file format. Use the metaImageIO from ITK.

class GateMHDImage
{
public:

  GateMHDImage();
  ~GateMHDImage();

  void ReadHeader(std::string & filename);
  void ReadData(std::string filename, std::vector<float> & data);

  void ReadHeader_old(std::string & filename);
  void ReadData_old(std::string filename, std::vector<float> & data);

  void WriteHeader(std::string filename, GateImage * image, bool writeData=false);
  void WriteData(std::string filename, GateImage * image);

  void WriteHeader_old(std::string filename, GateImage * image);
  void WriteData_old(std::string filename, GateImage * image);

  std::vector<double> size;
  std::vector<double> spacing;
  std::vector<double> origin;

  //-----------------------------------------------------------------------------
protected:
  std::vector<std::string> tags;
  std::vector<std::string> values;

  void Print();
  void Read_3_values(std::string tag, double * v);
  int  Read_tag(std::string tag);
  void Check_tag_value(std::string tag, std::string value);
  void EraseWhiteSpaces(std::string & s);
  void GetRawFilename(std::string filename, std::string & f, bool keepFolder);

};

#endif 



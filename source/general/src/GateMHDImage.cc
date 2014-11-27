/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


/*! \file
  \brief a 3D image
*/

#ifndef __GATEMHDIMAGE_CC__
#define __GATEMHDIMAGE_CC__

// std
#include <iomanip>
#include <sstream>
#include <iostream>

// gate
#include "GateMHDImage.hh"
#include "GateImageT.hh"
#include "GateMiscFunctions.hh"
#include "GateMachine.hh"

//-----------------------------------------------------------------------------
GateMHDImage::GateMHDImage() {
  tags.clear();
  values.clear();
  size.resize(3);
  spacing.resize(3);
  origin.resize(3);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateMHDImage::~GateMHDImage() {
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateMHDImage::ReadHeader(std::string & filename)
{
  GateMessage("Image",5,"GateMHDImage::ReadMHD " << filename << G4endl);
  //std::cout << "*** WARNING *** The mhd reader is experimental... (itk version)" << std::endl;

  MetaImage m_MetaImage;
  if(!m_MetaImage.Read(filename.c_str(), false)) {
    GateError("MHD File cannot be read: " << filename << std::endl);
  }

  if (m_MetaImage.NDims() != 3) {
    GateError("MHD File <" << filename << "> is not 3D but " << m_MetaImage.NDims() << "D, abort." << std::endl);
  }

  for(int i=0; i<m_MetaImage.NDims(); i++) {
    size[i] = m_MetaImage.DimSize(i);
    spacing[i] = m_MetaImage.ElementSpacing(i);
    origin[i] = m_MetaImage.Position(i);
  }

  transform.resize(9);
  for(int i=0; i<9; i++) { // 3 x 3 matrix
    transform[i] = m_MetaImage.TransformMatrix()[i];
  }
  //  Print();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateMHDImage::Print()
{
  std::cout << "Tags read = " << tags.size() << std::endl
            << "size = " << size[0] << " " << size[1] << " " << size[2] << std::endl
            << "spacing = " << spacing[0] << " " << spacing[1] << " " << spacing[2] << std::endl
            << "origin = " << origin[0] << " " << origin[1] << " " << origin[2] << std::endl;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateMHDImage::Read_3_values(std::string tag, double * v)
{
  int position = Read_tag(tag);
  std::istringstream is(values[position]);
  std::string s;
  is >> s; v[0] = atof(s.c_str()); // X
  is >> s; v[1] = atof(s.c_str()); // Y
  is >> s; v[2] = atof(s.c_str()); // Z
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
int GateMHDImage::Read_tag(std::string tag)
{
  std::vector<std::string>::iterator it = std::find(tags.begin(), tags.end(), tag);
  if (it == tags.end()) {
    GateError("Error while reading mhd header. I cannot find the tag '" << tag << "'");
  }
  int position = it - tags.begin();
  return position;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateMHDImage::Check_tag_value(std::string tag, std::string value)
{
  int position = Read_tag(tag);
  EraseWhiteSpaces(value);
  std::string v = values[position];
  EraseWhiteSpaces(v);
  if (v != value) {
    GateError("Error while reading mhd header. Tag '" << tag << "' should be '"
              << value << "' but I read '" << v << "'");
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateMHDImage::EraseWhiteSpaces(std::string & s)
{
  s.erase (std::remove(s.begin(), s.end(), ' '), s.end());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateMHDImage::GetRawFilename(std::string filename, std::string & f, bool keepFolder)
{
  unsigned int position;
  if (!keepFolder) {
    position = filename.find_last_of("/");
    filename = filename.substr(position+1, filename.size());
  }
  position = filename.find_last_of(".");
  filename = filename.substr(0,position+1);
  f = filename+"raw";
}
//-----------------------------------------------------------------------------


#endif

/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
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
GateMHDImage::GateMHDImage()
{
  tags.clear();
  values.clear();
  size.resize(3);
  spacing.resize(3);
  origin.resize(3);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
GateMHDImage::~GateMHDImage()
{
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateMHDImage::ReadHeader(std::string & filename)
{
  GateMessage("Image", 5, "GateMHDImage::ReadMHD " << filename << Gateendl);


  MetaImage m_MetaImage;
  if (!m_MetaImage.Read(filename.c_str(), false))
    {
      GateError("MHD File cannot be read: " << filename << Gateendl);
    }

  if (m_MetaImage.NDims() != 3)
    {
      GateError("MHD File <" << filename << "> is not 3D but " << m_MetaImage.NDims() << "D, abort.\n");
    }

  for (int i = 0; i < m_MetaImage.NDims(); i++)
    {
      size[i] = m_MetaImage.DimSize(i);
      // NOTE: The spacing and origin of MHD header files are (always?) rounded at 6 digits.
      // Thus rounding the spacing and the origin at 6 digits is mandatory for MHD images in order to keep original MHD header precision and avoid errors.
      spacing[i] = round_to_digits(m_MetaImage.ElementSpacing(i),6);
      origin[i]  = round_to_digits(m_MetaImage.Position(i),6);
    }

  transform.resize(9);
  for (int i = 0; i < 9; i++)
    { // 3 x 3 matrix
      transform[i] = m_MetaImage.TransformMatrix()[i];
    }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateMHDImage::Print()
{
  std::cout << "Tags read = " << tags.size() << Gateendl<< "size = " << size[0] << " " << size[1] << " " << size[2] << Gateendl
            << "spacing = " << spacing[0] << " " << spacing[1] << " " << spacing[2] << Gateendl
            << "origin = " << origin[0] << " " << origin[1] << " " << origin[2] << Gateendl;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateMHDImage::Read_3_values(std::string tag, double * v)
{
  int position = Read_tag(tag);
  std::istringstream is(values[position]);
  std::string s;
  is >> s;
  v[0] = atof(s.c_str()); // X
  is >> s;
  v[1] = atof(s.c_str()); // Y
  is >> s;
  v[2] = atof(s.c_str()); // Z
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
int GateMHDImage::Read_tag(std::string tag)
{
  std::vector<std::string>::iterator it = std::find(tags.begin(), tags.end(), tag);
  if (it == tags.end())
    {
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
  if (v != value)
    {
      GateError("Error while reading mhd header. Tag '" << tag << "' should be '" << value << "' but I read '" << v << "'");
    }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateMHDImage::EraseWhiteSpaces(std::string & s)
{
  s.erase(std::remove(s.begin(), s.end(), ' '), s.end());
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateMHDImage::GetRawFilename(std::string filename,
                                  std::string & f,
                                  bool keepFolder,
                                  bool changeExtension)
{
  unsigned int position;
  if (!keepFolder)
    {
      position = filename.find_last_of("/");
      filename = filename.substr(position + 1, filename.size());
    }
  position = filename.find_last_of(".");
  filename = filename.substr(0, position + 1);

  if (changeExtension == false)
    {
      f = filename + "raw";
    }
  else
    {
      f = filename + "sin";
    }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
double GateMHDImage::round_to_digits(double value, int digits)
{
  if (value == 0.0)
    return 0.0;

  double factor (pow(10.0, digits - ceil(log10(fabs(value)))));
  return round(value * factor) / factor;
}
//-----------------------------------------------------------------------------

#endif

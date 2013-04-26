/*----------------------
   GATE version name: gate_v6

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

#include "G4ThreeVector.hh"
#include <iomanip>
#include <sstream>
#include <iostream>

#include "GateImage.hh"
#include "GateMiscFunctions.hh"
#include "GateMachine.hh"
#include "GateMHDImage.hh"

// Include for mhd reader (extracted from ITK)
#include "metaObject.h"
#include "metaImage.h"

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
void GateMHDImage::ReadHeader_old(std::string & filename)
{
  GateMessage("Image",5,"GateMHDImage::ReadMHD " << filename << G4endl);
  //  std::cout << "*** WARNING *** The mhd reader is experimental..." << std::endl;

  std::ifstream is;
  is.open(filename.c_str(), std::ios::in);
  if ( is.fail() ) {
    GateError("Cannot open file '"<< filename << "'");
  }

  tags.clear();
  values.clear();
  while (is)  {
    std::string tag;
    is >> tag; // tag
    std::string s;
    is >> s; // '='
    char c[1024];
    is.getline(c, 1024);
    std::string d(c);
    tags.push_back(tag);
    values.push_back(d);
  }

  // Get information
  Check_tag_value("ObjectType", "Image");
  Check_tag_value("NDims", "3");
  Check_tag_value("BinaryData", "True");
  Check_tag_value("BinaryDataByteOrderMSB", "False");
  Check_tag_value("CompressedData", "False");
  Check_tag_value("TransformMatrix", "1 0 0 0 1 0 0 0 1");
 
  Read_3_values("ElementSpacing", &spacing[0]);
  Read_3_values("Offset", &origin[0]);
  Read_3_values("DimSize", &size[0]);
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
void GateMHDImage::ReadData(std::string filename, std::vector<float> & data)
{
  MetaImage m_MetaImage;
  if(!m_MetaImage.Read(filename.c_str(), true)) {
    GateError("MHD File cannot be read: " << filename << std::endl);
  }
  
  if (m_MetaImage.NDims() != 3) {
    GateError("MHD File <" << filename << "> is not 3D but " << m_MetaImage.NDims() << "D, abort." << std::endl);
  }

  // Convert to Float
  m_MetaImage.ConvertElementDataToIntensityData(MET_FLOAT);

  // Set data
  int len = size[0] * size[1] * size[2];
  data.assign((float*)(m_MetaImage.ElementData()), (float*)(m_MetaImage.ElementData()) + len);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateMHDImage::ReadData_old(std::string filename, std::vector<float> & data)
{
  typedef signed short int PixelType;
  
  // find filename 
  int p = Read_tag("ElementDataFile");
  std::string s = values[p];
  EraseWhiteSpaces(s);

  // build filename 
  std::string f;
  unsigned int position = filename.find_last_of("/");
  filename = filename.substr(0,position+1);
  s = filename+s;
  std::ifstream is;
  OpenFileInput(s, is);
  
  // Read data
  int nbOfValues = size[0] * size[1] * size[2];
  std::vector<PixelType> temp(nbOfValues);
  data.resize(nbOfValues);
  is.read((char*)(&(temp[0])), nbOfValues*sizeof(PixelType));
  for(unsigned int i=0; i<temp.size(); i++) {
    data[i] = (float)temp[i];
  }

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


//-----------------------------------------------------------------------------
void GateMHDImage::WriteHeader(std::string filename, GateImage * image, bool writeData)
{
  MetaImage m_MetaImage(image->GetResolution().x(), 
                        image->GetResolution().y(), 
                        image->GetResolution().z(), 
                        image->GetVoxelSize().x(), 
                        image->GetVoxelSize().y(), 
                        image->GetVoxelSize().z(), MET_FLOAT);
  std::string headName = filename;
  std::string dataName;
  GetRawFilename(filename, dataName, false);
  double p[3];
  p[0] = image->GetOrigin().x();
  p[1] = image->GetOrigin().y();
  p[2] = image->GetOrigin().z();
  m_MetaImage.Position(p);
  
  // Transform
  m_MetaImage.TransformMatrix(&image->GetTransformMatrix()[0]); // FIXME

  if (writeData) {
    m_MetaImage.ElementData(&(image->begin()[0]), false); // true = autofree
    m_MetaImage.Write(headName.c_str(), dataName.c_str());
  }
  else {
    m_MetaImage.Write(headName.c_str(), dataName.c_str(), false);
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateMHDImage::WriteHeader_old(std::string filename, GateImage * image)
{
  // Open file
  std::ofstream os;
  os.open(filename.c_str(), std::ios::out);
  if ( os.fail() ) {
    GateError("Cannot open file '"<< filename << "'");
  }
  // Create filename for raw
  std::string f;
  GetRawFilename(filename, f, false);

  // Write header
  os << "ObjectType = Image" << std::endl
     << "NDims = 3" << std::endl
     << "BinaryData = True" << std::endl
     << "BinaryDataByteOrderMSB = False" << std::endl
     << "CompressedData = False" << std::endl
     << "TransformMatrix = 1 0 0 0 1 0 0 0 1" << std::endl
     << "Offset = " << image->GetOrigin().x() << " " 
     << image->GetOrigin().y() << " " << image->GetOrigin().z() << std::endl
     << "CenterOfRotation = 0 0 0" << std::endl
     << "AnatomicalOrientation = RAI" << std::endl
     << "ElementSpacing = " << image->GetVoxelSize().x() << " " 
     << image->GetVoxelSize().y() << " " << image->GetVoxelSize().z() << std::endl
     << "DimSize = " << image->GetResolution().x() << " " 
     << image->GetResolution().y() << " " << image->GetResolution().z() << std::endl
     << "ElementType = MET_FLOAT" << std::endl
     << "ElementDataFile = " << f << std::endl;
  os.close();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateMHDImage::WriteData(std::string filename, GateImage * image)
{
  WriteHeader(filename, image, true);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateMHDImage::WriteData_old(std::string filename, GateImage * image)
{
 // Create filename for raw
  std::string f;
  GetRawFilename(filename, f, true); 
  std::ofstream os;
  OpenFileOutput(f, os);
  int nbOfValues = image->GetNumberOfValues();
  os.write((char*)(&(image->begin()[0])), nbOfValues*sizeof(float));
}
//-----------------------------------------------------------------------------


#endif 


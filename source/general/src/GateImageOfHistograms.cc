/*----------------------
  GATE version name: gate_v7

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
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
GateImageOfHistograms::GateImageOfHistograms(std::string dataTypeName):GateImage()
{
  SetHistoInfo(0,0,0);
  mDataTypeName = dataTypeName;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateImageOfHistograms::~GateImageOfHistograms()
{
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateImageOfHistograms::SetHistoInfo(int n, double min, double max)
{
  nbOfBins = n;
  minValue = min;
  maxValue = max;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateImageOfHistograms::Allocate()
{
  sizeX = resolution.x();
  sizeY = resolution.y();
  sizeZ = resolution.z();

  // Allocate full vector
  try {
    if (mDataTypeName == "double")
      dataDouble.resize(nbOfValues * nbOfBins); // FIXME To change for sparse allocation
    else if (mDataTypeName == "int")
      dataInt.resize(nbOfValues * nbOfBins); // FIXME To change for sparse allocation
    else
      dataFloat.resize(nbOfValues * nbOfBins); // FIXME To change for sparse allocation
  }
  catch(std::bad_alloc& ba) {
    double n = nbOfValues * nbOfBins;
    if (mDataTypeName == "double") n = n*sizeof(double);
    if (mDataTypeName == "int")    n = n*sizeof(int);
    else n = n*sizeof(float);
    GateError("Cannot allocate ImageOfHistogram, too much memory needed: " << n/(1024.0*1024.0) << " MB.\n"
              << " The image size is " << sizeX << "x" << sizeY << "x" << sizeZ
              << " and nbOfBins is " << nbOfBins);
  }

  // FIXME : Sparse
  /*
    mHistoData.resize(nbOfValues);
    // Could not allocate this way for 3D image -> too long !!
    for(int i=0; i<nbOfValues; i++) {
    // Create TH1D with no names (to save memory)
    //DD(i);
    mHistoData[i] = new TH1D("","", nbOfBins, minValue, maxValue);
    }
  */

  // Set to zero
  Reset();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateImageOfHistograms::UpdateSizesFromResolutionAndVoxelSize() {
  GateImage::UpdateSizesFromResolutionAndVoxelSize();
  // Store sizes in integer
  sizeX = resolution.x();
  sizeY = resolution.y();
  sizeZ = resolution.z();
  sizePlane = sizeX*sizeY;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateImageOfHistograms::UpdateSizesFromResolutionAndHalfSize() {
  GateImage::UpdateSizesFromResolutionAndHalfSize();
  // Store sizes in integer
  sizeX = resolution.x();
  sizeY = resolution.y();
  sizeZ = resolution.z();
  sizePlane = sizeX*sizeY;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateImageOfHistograms::Reset()
{
  if (mDataTypeName == "double")
    fill(dataDouble.begin(), dataDouble.end(), 0.0);
  else if (mDataTypeName == "float")
    fill(dataFloat.begin(), dataFloat.end(), 0.0);
  else
    fill(dataInt.begin(), dataInt.end(), 0);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateImageOfHistograms::Deallocate()
{
  // this thing frees the data memory, while keeping the rest intact. USE WITH CAUTION!
  if (mDataTypeName == "double")
    std::vector<double>().swap(dataDouble);
  else if (mDataTypeName == "float")
    std::vector<float>().swap(dataFloat);
  else
    std::vector<unsigned int>().swap(dataInt);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
long GateImageOfHistograms::GetIndexFromPixelIndex(int i, int j, int k)
{
  return (i + j*sizeX + k*sizePlane);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateImageOfHistograms::ComputeTotalOfCountsImageDataFloat(std::vector<float> & output)
{
  output.resize(nbOfValues);
  std::fill(output.begin(), output.end(), 0.0);
  unsigned long index_image = 0;
  unsigned long index_data = 0;
  for(unsigned int k=0; k<sizeZ; k++) {
    for(unsigned int j=0; j<sizeY; j++) {
      for(unsigned int i=0; i<sizeX; i++) {
        for(unsigned int l=0; l<nbOfBins; l++) {
          output[index_image] += dataFloat[index_data];
          index_data++;
        }
        index_image++;
      }
    }
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateImageOfHistograms::ComputeTotalOfCountsImageDataDouble(std::vector<double> & output)
{
  output.resize(nbOfValues);
  std::fill(output.begin(), output.end(), 0.0);
  unsigned long index_image = 0;
  unsigned long index_data = 0;
  for(unsigned int k=0; k<sizeZ; k++) {
    for(unsigned int j=0; j<sizeY; j++) {
      for(unsigned int i=0; i<sizeX; i++) {
        for(unsigned int l=0; l<nbOfBins; l++) {
          output[index_image] += dataDouble[index_data];
          index_data++;
        }
        index_image++;
      }
    }
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateImageOfHistograms::AddValueFloat(const int & index, TH1D * h, const double scale=1.0)
{
  int index_data = index*nbOfBins;
  for(unsigned int i=1; i<=nbOfBins; i++) {
    // +1 because TH1D start at 1, and end at index=size
    dataFloat[index_data] += h->GetBinContent(i)*scale;
    index_data++;
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateImageOfHistograms::AddValueDouble(const int & index, TH1D * h, const double scale=1.0)
{
  int index_data = index*nbOfBins;
  for(unsigned int i=1; i<=nbOfBins; i++) {
    // +1 because TH1D start at 1, and end at index=size
    dataDouble[index_data] += h->GetBinContent(i)*scale;
    index_data++;
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateImageOfHistograms::AddValueDouble(const int & index, const int &bin, const double value=1.0)
{
  long index_data = index*nbOfBins+bin;
  dataDouble[index_data] += value;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateImageOfHistograms::SetValueDouble(const int & index, const int &bin, const double value=1.0)
{
  long index_data = index*nbOfBins+bin;
  dataDouble[index_data] = value;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
double GateImageOfHistograms::GetValueDouble(const int & index, const int &bin)
{
  long index_data = index*nbOfBins+bin;
  return dataDouble[index_data];
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateImageOfHistograms::AddValueInt(const int & index, const int & bin, const unsigned int value=1)
{
  long index_data = index*nbOfBins+bin;
  dataInt[index_data] += value;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateImageOfHistograms::Read(G4String filename)
{
  std::vector<float> input;//put mhd imagedata here

  {//start metaImage scope. setup all metadata. separate scope to save memory
    MetaImage m_MetaImage;
    m_MetaImage.AddUserField("HistoMin", MET_FLOAT_ARRAY, 1);
    m_MetaImage.AddUserField("HistoMax", MET_FLOAT_ARRAY, 1);

    if (!m_MetaImage.Read(filename.c_str(), true)) {
      GateError("MHD File cannot be read: " << filename << std::endl);
    }

    // Dimension must be 4
    if (m_MetaImage.NDims() != 4) {
      GateError("MHD ImageOfHistogram  <" << filename << "> is not 4D but "
                << m_MetaImage.NDims() << "D, abort." << std::endl);
    }

    // Get image parameters
    for(int i=0; i<3; i++) {
      resolution[i] = m_MetaImage.DimSize(i);
      voxelSize[i] = m_MetaImage.ElementSpacing(i);
      origin[i] = m_MetaImage.Position(i);
    }
    nbOfBins = m_MetaImage.DimSize(3);

    std::vector<double> transform;
    transform.resize(16);
    for(int i=0; i<16; i++) { // 4 x 4 matrix
      transform[i] = m_MetaImage.TransformMatrix()[i];
    }

    // Convert mhd 4D matrix to 3D rotation matrix
    G4ThreeVector row_x, row_y, row_z;
    for(unsigned int i=0; i<3; i++) {
      row_x[i] = transform[i*4];
      row_y[i] = transform[i*4+1];
      row_z[i] = transform[i*4+2];
    }
    transformMatrix.setRows(row_x, row_y, row_z);
    if( !transformMatrix.row1().isNear(CLHEP::HepLorentzVector(row_x, 0.), 0.1) ||
        !transformMatrix.row2().isNear(CLHEP::HepLorentzVector(row_y, 0.), 0.1) ||
        !transformMatrix.row3().isNear(CLHEP::HepLorentzVector(row_z, 0.), 0.1) ) {
      GateError(filename << " contains a transformation which is not a rotation. "
                << "It is probably a flip and this is not handled.");
    }

    // Read info in mhd
    void * r = 0;
    r = m_MetaImage.GetUserField("HistoMin");
    if (r==0) {
      GateError("User field HistoMin not found in this mhd file : " << filename);
    }
    minValue = *static_cast<float*>(r);
    r = m_MetaImage.GetUserField("HistoMax");
    if (r==0) {
      GateError("User field HistoMax not found in this mhd file : " << filename);
    }
    maxValue = *static_cast<float*>(r);
    SetHistoInfo(nbOfBins, minValue, maxValue);

    // We need to shift to half a pixel to be coherent with Gate
    // coordinates system. Must be transformed because voxel size is
    // known before rotation and origin is after rotation.
    origin -= transformMatrix*(voxelSize/2.0);
    UpdateSizesFromResolutionAndVoxelSize();

    // Cast input data to float and put in input vector. this is done because metaImage nullpointer sucks.
    int len = resolution[0] * resolution[1] * resolution[2] * nbOfBins;
    input.assign((float*)(m_MetaImage.ElementData()), (float*)(m_MetaImage.ElementData()) + len);

  }//end metaImage scope. it does not exist anymore.

  //Set to correct order. This allocates dataFloat.
  ConvertPixelOrderToHXYZ(input, dataFloat);

  // Do NOT clear metaimages yourself, then the destructor will NOT deallocate datamemory!
  // m_MetaImage.Clear();

  // FIXME: this couldn't possibly work. Incase we were reading an image of doubles,
  // we'd already casted in the previous steps to float, which we even assume, so no
  // double image caant be read right now. Would only be usefull if we would continue
  // to update this image with small values, which I don't think will ever happen.
  /*if (mDataTypeName == "double") {
    dataDouble.resize(dataFloat.size());
    for(unsigned int i=0; i<dataDouble.size(); i++)
    dataDouble[i] = (double)dataFloat[i]; // convert float to double
    dataFloat.clear();
    }*/

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateImageOfHistograms::Write(G4String filename, const G4String & )
{
  // Filenames
  std::string extension = getExtension(filename);
  std::string baseName = removeExtension(filename);
  std::string headerName = filename;
  std::string rawName = baseName+".raw";

  // FIXME : FIRST version with FULL raw data ; next to change in
  // dynamic allocation, keep empty check extension type mhd ==> no
  // change to ioh (ImageOfHistogram) ?

  // Create a mhd header : 4D image with the 4th dimension is the histogram
  int * dimSize = new int[4];
  float * spacing = new float[4];
  dimSize[0] = GetResolution().x();
  dimSize[1] = GetResolution().y();
  dimSize[2] = GetResolution().z();
  dimSize[3] = nbOfBins; // histogram
  spacing[0] = GetVoxelSize().x();
  spacing[1] = GetVoxelSize().x();
  spacing[2] = GetVoxelSize().x();
  spacing[3] = 1; // Dummy

  // Always write in float rather than double to preserve memory, but
  // computation could be performed in double to prevent potential
  // rounding error.
  MetaImage m_MetaImage(4, dimSize, spacing, MET_FLOAT);
  m_MetaImage.AddUserField("HistoMin", MET_FLOAT_ARRAY, 1, &minValue); // unit: MeV for energy, ns for time
  m_MetaImage.AddUserField("HistoMax", MET_FLOAT_ARRAY, 1, &maxValue); // unit: MeV for energy, ns for time

  double p[4];
  // Gate convention: origin is the corner of the first pixel
  // MHD / ITK convention: origin is the center of the first pixel
  // -> Add a half pixel
  p[0] = GetOrigin().x() + GetVoxelSize().x()/2.0;
  p[1] = GetOrigin().y() + GetVoxelSize().y()/2.0;
  p[2] = GetOrigin().z() + GetVoxelSize().z()/2.0;
  p[3] = 0.; // needed to be defined as a unique value for the merging of several jobs
  m_MetaImage.Position(p);

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
  m_MetaImage.TransformMatrix(matrix);

  // Before writing convert from double to float
  double total = 0.0;
  if (mDataTypeName == "double") {
    dataFloat.resize(dataDouble.size());
    for(unsigned int i=0; i<dataDouble.size(); i++) {
      total += dataDouble[i];
      dataFloat[i] = (float)dataDouble[i]; // convert double to float
    }
    std::vector<double>().swap(dataDouble);
    //we swap with empty vector, thereby releasing memory of original dataDouble
    //once this new vector, with the contents of dataDouble, goes out of scope (after this line)
  }

  //output int to int
  if (mDataTypeName == "int") {
    // Change the order of the pixels : store on disk as XYZH.
    std::vector<unsigned int> t;
    ConvertPixelOrderToXYZH(dataInt, t);
    m_MetaImage.ElementData(&(t.begin()[0]), false); // true = autofree
    m_MetaImage.Write(headerName.c_str(), rawName.c_str());
  } else {
    //if not int, then write as float.
    m_MetaImage.AddUserField("TotalSum", MET_FLOAT_ARRAY, 1, &total);

    // Change the order of the pixels : store on disk as XYZH.
    std::vector<float> t;
    ConvertPixelOrderToXYZH(dataFloat, t);
    m_MetaImage.ElementData(&(t.begin()[0]), false); // true = autofree
    m_MetaImage.Write(headerName.c_str(), rawName.c_str());
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// This scales the Image, and it converts an Int Image into a Float Image.
//-----------------------------------------------------------------------------
void GateImageOfHistograms::Scale(double f)
{
  if (mDataTypeName == "double") {
    for(unsigned int i=0; i<dataDouble.size(); i++)
      dataDouble[i] = f * dataDouble[i];
  } else if (mDataTypeName == "int") { //cant scale and int
    dataFloat.resize(dataInt.size());
    for(unsigned int i=0; i<dataInt.size(); i++) {
      dataFloat[i] = f * (float)dataInt[i]; // convert int to float
    }
    mDataTypeName = "float";
    std::vector<unsigned int>().swap(dataInt);
    //we swap with empty vector, thereby releasing memory of original dataDouble
    //once this new vector, with the contents of dataDouble, goes out of scope (after this line)
  } else {
    for(unsigned int i=0; i<dataFloat.size(); i++)
      dataFloat[i] = f * dataFloat[i];
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
double GateImageOfHistograms::ComputeSum()
{
  double sum = 0.0;
  if (mDataTypeName == "double") {
    for(unsigned int i=0; i<dataDouble.size(); i++)
      sum += dataDouble[i];
  }
  else {
    for(unsigned int i=0; i<dataFloat.size(); i++)
      sum += dataFloat[i];
  }
  return sum;
}
//-----------------------------------------------------------------------------

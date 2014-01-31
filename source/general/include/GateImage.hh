/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


/*!
  \class GateImage
  \ingroup data_structures
  \author thibault.frisson@creatis.insa-lyon.fr
  laurent.guigues@creatis.insa-lyon.fr
  david.sarrut@creatis.insa-lyon.fr
*/

#ifndef __GateImage_hh__
#define __GateImage_hh__

#include <globals.hh>
#include <G4ThreeVector.hh>
#include <G4RotationMatrix.hh>
#include <vector>

#include "GateAnalyzeHeader.hh"
#include "GateInterfileHeader.hh"
#include "GateMessageManager.hh"

class G4String;


/// \brief 3D images of PixelType values
class GateImage
{
public:

  typedef float PixelType;
  typedef std::vector<PixelType>::iterator iterator;
  typedef std::vector<PixelType>::const_iterator const_iterator;

  GateImage();
  ~GateImage();

  /// Sets the image dimensions from resolution and half size
  void SetResolutionAndHalfSize(G4ThreeVector r, G4ThreeVector h);
  void SetResolutionAndHalfSize(G4ThreeVector r, G4ThreeVector h, G4ThreeVector position);

  /// Sets the image dimensions from resolution and voxel size
  void SetResolutionAndVoxelSize(G4ThreeVector r, G4ThreeVector v);
  void SetResolutionAndVoxelSize(G4ThreeVector r, G4ThreeVector v, G4ThreeVector position);

  /// Allocates the data
  void Allocate();

  /// Returns the size of the image
  inline G4ThreeVector GetSize()           const { return size; }

  /// Returns the half size of the image
  inline G4ThreeVector GetHalfSize()       const { return halfSize; }

  /// Returns the resolution of the image (number of voxels in x,y and z directions, i.e. number of columns, lines and planes)
  inline G4ThreeVector GetResolution()     const { return resolution; }

  /// Returns the voxels size
  inline G4ThreeVector GetVoxelSize()      const { return voxelSize; }

  /// Returns the volume of the voxels
  inline G4double      GetVoxelVolume()    const { return voxelVolume; }

  /// Returns the number of values in the image (i.e. the number of voxels)
  inline int           GetNumberOfValues() const { return nbOfValues; }

  /// Returns the line size
  inline int           GetLineSize()       const { return lineSize; }

  /// Returns the plane size
  inline int           GetPlaneSize()      const { return planeSize; }

  /// Returns origin
  inline G4ThreeVector GetOrigin()     const { return origin; }
  void SetOrigin(G4ThreeVector o) { origin = o; }

  const G4RotationMatrix & GetTransformMatrix() const { return transformMatrix; }
  inline void SetTransformMatrix(const G4RotationMatrix &transMatrix) { transformMatrix = transMatrix; }

  bool HasSameResolutionThan(const GateImage & image) const;
  bool HasSameResolutionThan(const GateImage * pImage) const;

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

  PixelType GetMinValue();

  inline PixelType GetOutsideValue()   { return mOutsideValue; } 
  inline void SetOutsideValue( PixelType v ) { mOutsideValue=v; }

  // Transformations between systems of coordinates 

  /// Returns the voxel's coordinates from its index : OK
  G4ThreeVector GetCoordinatesFromIndex(int index) const;
  /// Returns the voxel's index from its coordinates : OK
  inline int GetIndexFromCoordinates(const G4ThreeVector & coords) const
  { return (int)(coords.x()+coords.y()*lineSize+coords.z()*planeSize); }


  /// Returns the position of the center of the voxel from the coordinates : OK 
  G4ThreeVector GetVoxelCenterFromCoordinates(G4ThreeVector coords) const;
  /// Returns the position of the center of the voxel from the index : OK 
  G4ThreeVector GetVoxelCenterFromIndex(int index) const 
  { return GetVoxelCenterFromCoordinates(GetCoordinatesFromIndex(index)); }

  /// Returns the position of the corner of the voxel from the coordinates : OK
  G4ThreeVector GetVoxelCornerFromCoordinates(G4ThreeVector coords) const;
  /// Returns the position of the corner of the voxel from the index : OK
  G4ThreeVector GetVoxelCornerFromIndex(int index) const
  { return GetVoxelCornerFromCoordinates(GetCoordinatesFromIndex(index)); }

  /// Returns the x of the corner of the voxel given the x coordinate
  G4double GetXVoxelCornerFromXCoordinate(int i) const { return i * voxelSize.x() - halfSize.x(); }
  /// Returns the y of the corner of the voxel given the y coordinate
  G4double GetYVoxelCornerFromYCoordinate(int j) const { return j * voxelSize.y() - halfSize.y(); }
  /// Returns the z of the corner of the voxel given the z coordinate
  G4double GetZVoxelCornerFromZCoordinate(int k) const { return k * voxelSize.z() - halfSize.z(); }

  G4int            GetVoxelNx()        { return m_voxelNx; };
  G4int            GetVoxelNy()        { return m_voxelNy; };
  G4int            GetVoxelNz()        { return m_voxelNz; };

  // Returns the index from the position : OK 
  int GetIndexFromPosition(const G4ThreeVector& position) const;  
  int GetIndexFromPositionAndDirection(const G4ThreeVector& position, const G4ThreeVector& direction) const;  
  int GetIndexFromPostPositionAndDirection(const G4ThreeVector& position, const G4ThreeVector& direction) const;  
  int GetIndexFromPostPosition(const G4ThreeVector& pre, const G4ThreeVector& post) const;  
  int GetIndexFromPrePosition(const G4ThreeVector& pre, const G4ThreeVector& post) const;  
  int GetIndexFromPostPosition(const double t, const double pret, const double postt, const double resolutiont) const;
  int GetIndexFromPrePosition(const double t, const double pret, const double postt, const double resolutiont) const;

  // Returns the (integer) coordinates of the voxel in which the point is : OK
  G4ThreeVector GetCoordinatesFromPosition(const G4ThreeVector & position);
  // Returns the (integer) coordinates of the voxel in which the point is : OK
  void GetCoordinatesFromPosition(const G4ThreeVector & position, int& i, int& j, int& k);

  G4ThreeVector GetNonIntegerCoordinatesFromPosition(G4ThreeVector position);

  enum ESide {kUndefined,kPX,kMX,kPY,kMY,kPZ,kMZ};
  ESide GetSideFromPointAndCoordinate(const G4ThreeVector & p, const G4ThreeVector & c);
  PixelType GetNeighborValueFromCoordinate(const ESide & side, const G4ThreeVector & coord);

  // iterators 
  iterator begin() { return data.begin(); }
  iterator end()   { return data.end(); }
  const_iterator begin() const { return data.begin(); }
  const_iterator end() const  { return data.end(); }

  // IO
  /// Writes the image to a file with comment (the format is detected automatically)
  void Write(G4String filename, const G4String & comment = "");
  /// Reads the image from a file (the format is detected automatically)
  void Read(G4String filename);
  void MergeDataByAddition(G4String filename);
  /// Displays info about the image to standard output
  void PrintInfo();

  //-----------------------------------------------------------------------------
protected:
  int nbOfValues;
  G4ThreeVector size;
  G4ThreeVector halfSize;
  G4ThreeVector resolution;
  G4ThreeVector voxelSize;
  G4ThreeVector origin;
  G4RotationMatrix transformMatrix;
  G4double      voxelVolume;
  G4double      kCarTolerance;
  G4ThreeVector halfSizeMinusVoxelCenter;
  int planeSize;
  int lineSize;
  std::vector<PixelType> data;
  G4ThreeVector  mPosition;

  G4int                          m_voxelNx;
  G4int                          m_voxelNy;
  G4int                          m_voxelNz;

  PixelType mOutsideValue;

  void ReadVox(G4String filename);
  void ReadAscii(G4String filename);
  void ReadVox2(std::ifstream & is);
  void ReadVox3(std::ifstream & is);
  
  void EraseWhiteSpaces(std::string & s);
  void ReadAnalyze(G4String filename);
  void Read_mhd_3_values(std::ifstream & is, double * values);
  void Read_mhd_tag(std::ifstream & is, std::string tag);
  void Read_mhd_tag_check_value(std::ifstream & is, std::string tag, std::string value);
  void Read_mhd_tag_check_value(std::vector<std::string> & tags, 
                                std::vector<std::string> & values, 
                                std::string tag, std::string value);
  void ReadMHD(G4String filename);
  void ReadInterfile(G4String fileName);

  void WriteBin(std::ofstream & os);
  void WriteAscii(std::ofstream & os, const G4String & comment);
  void WriteVox(std::ofstream & os);
  void WriteAnalyzeHeader(G4String filename);
  void WriteRoot(G4String filename);
  void WriteMHD(std::string filename);

  void UpdateNumberOfValues();
  void UpdateSizesFromResolutionAndHalfSize();
  void UpdateSizesFromResolutionAndVoxelSize();

  // data for root output
  void UpdateDataForRootOutput();
  int mRootHistoDim;
  int mRootHistoBinxNb;
  double mRootHistoBinxLow;
  double mRootHistoBinxUp;
  double mRootHistoBinxSize;
  int mRootHistoBinyNb;
  double mRootHistoBinyLow;
  double mRootHistoBinyUp;
  double mRootHistoBinySize;
  int mRootHistoBinzNb;
  double mRootHistoBinzLow;
  double mRootHistoBinzUp;
  double mRootHistoBinzSize;  
};

#endif 



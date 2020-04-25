/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/


/*!
  \class GateVImage
  \ingroup data_structures
  \author thibault.frisson@creatis.insa-lyon.fr
  laurent.guigues@creatis.insa-lyon.fr
  david.sarrut@creatis.insa-lyon.fr
*/

#ifndef __GATEVIMAGE_HH__
#define __GATEVIMAGE_HH__

#include <globals.hh>
#include <G4ThreeVector.hh>
#include <G4RotationMatrix.hh>
#include <vector>

#include "GateAnalyzeHeader.hh"
#include "GateInterfileHeader.hh"
#include "GateMessageManager.hh"

class G4String;


/// \brief 3D images of PixelType values
class GateVImage
{
public:


  GateVImage();
  virtual ~GateVImage();

  /// Sets the image dimensions from resolution and half size
  void SetResolutionAndHalfSize(G4ThreeVector r, G4ThreeVector h);
  void SetResolutionAndHalfSize(G4ThreeVector r, G4ThreeVector h, G4ThreeVector position);

  /// Sets the image dimensions from resolution and voxel size
  void SetResolutionAndVoxelSize(G4ThreeVector r, G4ThreeVector v);
  void SetResolutionAndVoxelSize(G4ThreeVector r, G4ThreeVector v, G4ThreeVector position);
  
  /// Sets the image dimensions from resolution and half size for cylindrical symmetry 
  void SetResolutionAndHalfSizeCylinder(G4ThreeVector r, G4ThreeVector h, G4ThreeVector position);
  void SetResolutionAndHalfSizeCylinder(G4ThreeVector r, G4ThreeVector h);

  /// Allocates the data
  virtual void Allocate() = 0;

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

  bool HasSameResolutionThan(const GateVImage & image) const;
  bool HasSameResolutionThan(const GateVImage * pImage) const;

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

  G4int GetVoxelNx() { return m_voxelNx; };
  G4int GetVoxelNy() { return m_voxelNy; };
  G4int GetVoxelNz() { return m_voxelNz; };

  // Returns the index from the position : OK
  int GetIndexFromPosition(const G4ThreeVector& position) const;
  int GetIndexFromPositionAndDirection(const G4ThreeVector& position, const G4ThreeVector& direction) const;
  int GetIndexFromPostPositionAndDirection(const G4ThreeVector& position, const G4ThreeVector& direction) const;
  int GetIndexFromPostPosition(const G4ThreeVector& pre, const G4ThreeVector& post) const;
  int GetIndexFromPrePosition(const G4ThreeVector& pre, const G4ThreeVector& post) const;
  int GetIndexFromPositionCylindricalCS(const G4ThreeVector& position) const; 
  
  int GetIndexFromPostPosition(const double t, const double pret, const double postt, const double resolutiont) const;
  int GetIndexFromPrePosition(const double t, const double pret, const double postt, const double resolutiont) const;

  // Returns the (integer) coordinates of the voxel in which the point is : OK
  G4ThreeVector GetCoordinatesFromPosition(const G4ThreeVector & position);
  // Returns the (integer) coordinates of the voxel in which the point is : OK
  void GetCoordinatesFromPosition(const G4ThreeVector & position, int& i, int& j, int& k);

  G4ThreeVector GetNonIntegerCoordinatesFromPosition(G4ThreeVector position);

  enum ESide {kUndefined,kPX,kMX,kPY,kMY,kPZ,kMZ};
  ESide GetSideFromPointAndCoordinate(const G4ThreeVector & p, const G4ThreeVector & c);


  // IO

  /// Writes the image to a file with comment (the format is detected automatically)
  virtual void Write(G4String filename, const G4String & comment = "") = 0;

  /// Reads the image from a file (the format is detected automatically)
  virtual void Read(G4String filename) = 0;

  /// Displays info about the image to standard output
  virtual void PrintInfo() = 0;

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
  G4ThreeVector  mPosition;

  G4int                          m_voxelNx;
  G4int                          m_voxelNy;
  G4int                          m_voxelNz;

  void UpdateNumberOfValues();
  void UpdateSizesFromResolutionAndHalfSize();
  void UpdateSizesFromResolutionAndVoxelSize();

  void UpdateSizesFromResolutionAndHalfSizeCylinder();
  void UpdateSizesFromResolutionAndVoxelSizeCylinder();
  
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

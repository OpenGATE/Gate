/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*!
  \class  GateVoxelizedMass
  \author Thomas DESCHLER (thomas.deschler@iphc.cnrs.fr)
  \date	October 2015
 */

#ifndef GATEVOXELIZEDMASS_HH
#define GATEVOXELIZEDMASS_HH

#include <G4UnitsTable.hh>
#include <G4Box.hh>

#include "GateVImageActor.hh"
#include "GateVImageVolume.hh"
#include "GateActorManager.hh"
#include "GateImageWithStatistic.hh"


class GateVoxelizedMass
{
 public:

  GateVoxelizedMass();
  virtual ~GateVoxelizedMass() {}

  void Initialize(G4String, const GateVImage*);

  void UpdateImage(GateImageDouble*);

  std::vector<double> GetDoselMassVector();

  G4String GetVoxelMatName(int x, int y, int z);
  G4double GetVoxelMass(int x, int y, int z);
  G4double GetVoxelVolume();

  G4double GetDoselMass(int index);
  G4double GetDoselVolume();

  double  GetPartialVolumeWithSV     (int index,G4String);
  double  GetPartialMassWithSV       (int index,G4String);
  //double  GetPartialVolumeWithMatName(int index);
  double  GetPartialMassWithMatName  (int index);
  double  GetTotalVolume();
  int     GetNumberOfVolumes(int index);
  double  GetMaxDose(int index);

  void    SetEdep(int index,G4String,double);
  void    SetMaterialFilter   (G4String);
  void    SetVolumeFilter     (G4String);

  void    SetExternalMassImage(G4String);

 protected:

  bool IsLVParameterized(const G4LogicalVolume*);

  void GenerateVectors();
  void GenerateVoxels();
  void GenerateDosels(int index);

  std::pair<double,double> ParameterizedVolume(int index);
  std::pair<double,double> VoxelIteration(const G4VPhysicalVolume*,int, G4RotationMatrix, G4ThreeVector,int index);

  GateVImageVolume* imageVolume;
  const G4VPhysicalVolume* DAPV;
  const G4LogicalVolume* DALV;
  G4Box* doselSV;
  const G4Box* DABox;

  const GateVImage* mImage;
  GateImageDouble* mMassImage;

  double doselReconstructedTotalCubicVolume;
  double doselReconstructedTotalMass;

  std::pair<double,double> doselReconstructedData;

  std::vector<std::vector<std::pair<G4String,double> > > mCubicVolume;
  std::vector<std::vector<std::pair<G4String,double> > > mMass;
  std::vector<std::vector<std::pair<G4String,double> > > mEdep;

  //std::vector<double> doselReconstructedCubicVolume;
  std::vector<double> doselReconstructedMass;
  std::vector<double> doselMin;
  std::vector<double> doselMax;
  std::vector<double> doselExternalMass;

  double voxelCubicVolume;
  double mFilteredVolumeMass;
  double mFilteredVolumeCubicVolume;

  //std::vector<G4VSolid*> vectorSV;


  G4String mVolumeName;
  G4String mMassFile;
  G4String mMaterialFilter;
  G4String mVolumeFilter;

  bool mIsInitialized;
  bool mIsParameterised;
  bool mIsVecGenerated;
  bool mIsFilteredVolumeProcessed;
  bool mHasFilter;
  bool mHasExternalMassImage;
  bool mHasSameResolution;

  int seconds;
};

#endif

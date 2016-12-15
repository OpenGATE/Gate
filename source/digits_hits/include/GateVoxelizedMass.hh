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

  void Initialize(const G4String, const GateVImage*);

  virtual bool IsLVParameterized(const G4LogicalVolume* LV);
  virtual void GenerateVectors();
  virtual void GenerateVoxels();
  virtual void GenerateDosels(const int index);
  virtual std::pair<double,double> ParameterizedVolume(const int index);
  virtual std::pair<double,double> VoxelIteration(G4VPhysicalVolume* motherPV,const int Generation,G4RotationMatrix MotherRotation,G4ThreeVector MotherTranslation,const int index);
  std::vector<double> GetDoselMassVector();

  G4String GetVoxelMatName(int x, int y, int z);
  G4double GetVoxelMass(int x, int y, int z);
  G4double GetVoxelVolume();

  G4double GetDoselMass(const int index);
  G4double GetDoselVolume(const int index);

  double  GetPartialVolumeWithSV     (const int index,const G4String SVName);
  double  GetPartialMassWithSV       (const int index,const G4String SVName);
  double  GetPartialVolumeWithMatName(const int index);
  double  GetPartialMassWithMatName  (const int index);
  double  GetTotalVolume();
  int     GetNumberOfVolumes(const int index);
  double  GetMaxDose(const int index);
  void    SetEdep(const int index,const G4String SVName,const double Edep);
  void    SetMaterialFilter(const G4String MatName);
  void    SetVolumeFilter(const G4String MatName);
  void    SetExternalMassImage(const G4String extMassFile);

  void UpdateImage(GateImageDouble*);

 protected:

  GateVImageVolume* imageVolume;
  G4VPhysicalVolume* DAPV;
  G4LogicalVolume* DALV;
  G4Box* doselSV;
  G4Box* DABox;

  double doselReconstructedTotalCubicVolume;
  double doselReconstructedTotalMass;

  std::pair<double,double> doselReconstructedData;

  std::vector<std::vector<std::pair<G4String,double> > > mCubicVolume;
  std::vector<std::vector<std::pair<G4String,double> > > mMass;
  std::vector<std::vector<std::pair<G4String,double> > > mEdep;

  std::vector<double> doselReconstructedCubicVolume;
  std::vector<double> doselReconstructedMass;
  std::vector<double> doselMin;
  std::vector<double> doselMax;
  std::vector<double> doselExternalMass;

  double voxelCubicVolume;
  double mFilteredVolumeMass;
  double mFilteredVolumeCubicVolume;

  std::vector<G4VSolid*> vectorSV;

  const GateVImage* mImage;
  GateImageDouble mMassImage;

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

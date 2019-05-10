/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/


/*! \file
  \brief Implementation of GateVImageVolumeMessenger
*/
#include "GateVImageVolumeMessenger.hh"
#include "GateVImageVolume.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithADouble.hh"

//---------------------------------------------------------------------------
GateVImageVolumeMessenger::GateVImageVolumeMessenger(GateVImageVolume* volume)
  :
  GateVolumeMessenger(volume),
  pVImageVolume(volume)
{

  GateMessage("Volume",5,"GateVImageVolumeMessenger()\n");
  G4String dir = GetDirectoryName() + "geometry";
  //  G4cout<<dir<< Gateendl;

  G4String n = dir +"/setImage";
  pImageFileNameCmd = 0;
  pImageFileNameCmd = new G4UIcmdWithAString(n,this);
  pImageFileNameCmd->SetGuidance("Sets the name of the image file");

  n = dir +"/SetImage";
  pImageFileNameCmdDeprecated = 0;
  pImageFileNameCmdDeprecated = new G4UIcmdWithAString(n,this);
  pImageFileNameCmdDeprecated->SetGuidance("(DEPRECATED) Sets the name of the image file");

  // Disable this macro for the moment
  n = dir +"/setLabelToMaterialFile";
  pLabelToMaterialFileNameCmd = new G4UIcmdWithAString(n,this);
  pLabelToMaterialFileNameCmd->SetGuidance("Sets the name of the file containing the label to material correspondence");

  n = dir +"/setHUToMaterialFile";
  pHUToMaterialFileNameCmd = 0;
  pHUToMaterialFileNameCmd = new G4UIcmdWithAString(n,this);
  pHUToMaterialFileNameCmd->SetGuidance("Sets the name of the file containing the HU intervals to material correspondence");

  n = dir +"/SetHUToMaterialFile";
  pHUToMaterialFileNameCmdDeprecated = 0;
  pHUToMaterialFileNameCmdDeprecated = new G4UIcmdWithAString(n,this);
  pHUToMaterialFileNameCmdDeprecated->SetGuidance("Sets the name of the file containing the HU intervals to material correspondence");

  n = dir +"/setRangeToMaterialFile";
  pRangeMaterialFileNameCmd = 0;
  pRangeMaterialFileNameCmd = new G4UIcmdWithAString(n,this);
  pRangeMaterialFileNameCmd->SetGuidance("Sets the name of the file containing the intervals to material correspondence");

  n = dir +"/TranslateTheImageAtThisIsoCenter";
  pIsoCenterCmd = 0;
  pIsoCenterCmd = new G4UIcmdWith3VectorAndUnit(n,this);
  pIsoCenterCmd->SetGuidance("translate the image so that its center is at the given isocenter coordinate (given according to the 3D image coordinate, in mm)");

  n = dir +"/setRotationAroundPixelIsoCenter";
  pIsoCenterRotationFlagCmd = 0;
  pIsoCenterRotationFlagCmd = new G4UIcmdWithABool(n,this);
  pIsoCenterRotationFlagCmd->SetParameterName(n, false); // not omittable
  pIsoCenterRotationFlagCmd->SetGuidance("If a rotation is applied to the volume, consider around the pixel isocenter.");

  n = dir +"/setOrigin";
  pSetOriginCmd = 0;
  pSetOriginCmd = new G4UIcmdWith3VectorAndUnit(n,this);
  pSetOriginCmd->SetGuidance("Set the image origin (like in dicom).");

  n = dir +"/buildAndDumpDistanceTransfo";
  pBuildDistanceTransfoCmd = 0;
  pBuildDistanceTransfoCmd = new G4UIcmdWithAString(n,this);
  pBuildDistanceTransfoCmd->SetGuidance("Build and dump the distance transfo into the given filename.");

  n = dir +"/buildAndDumpLabeledImage";
  pBuildLabeledImageCmd = 0;
  pBuildLabeledImageCmd = new G4UIcmdWithAString(n,this);
  pBuildLabeledImageCmd->SetGuidance("Build and dump the image labeled according to the materials list. Give the filename.");

  n = dir +"/buildAndDumpDensityImage";
  pBuildDensityImageCmd = 0;
  pBuildDensityImageCmd = new G4UIcmdWithAString(n,this);
  pBuildDensityImageCmd->SetGuidance("Build and dump the density image (in g/cm3) according to the materials list. Give the filename.");

  n = dir +"/buildAndDumpMassImage";
  pBuildMassImageCmd = new G4UIcmdWithAString(n,this);
  pBuildMassImageCmd->SetGuidance("Build and dump the mass image (in g) according to the materials list. Give the filename.");

  n = dir +"/enableBoundingBoxOnly";
  pDoNotBuildVoxelsCmd = 0;
  pDoNotBuildVoxelsCmd = new G4UIcmdWithABool(n,this);
  pDoNotBuildVoxelsCmd->SetGuidance("Only build the bounding box (no voxels !), for visualization purpose only.");

  n = dir +"/setMaxOutOfRangeFraction";
  pSetMaxOutOfRangeFractionCmd = new G4UIcmdWithADouble(n,this);
  pSetMaxOutOfRangeFractionCmd->SetGuidance("Maximum fraction (number between 0.0 and 1.0) of voxels that have a HU value out of the range of the materials table.");
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
GateVImageVolumeMessenger::~GateVImageVolumeMessenger()
{
  GateMessage("Volume",5,"~GateVImageVolumeMessenger()\n");

  delete pImageFileNameCmd;
  delete pImageFileNameCmdDeprecated;
  delete pLabelToMaterialFileNameCmd;
  delete pHUToMaterialFileNameCmd;
  delete pRangeMaterialFileNameCmd;
  delete pBuildDistanceTransfoCmd;
  delete pIsoCenterCmd;
  delete pSetOriginCmd;
  delete pBuildLabeledImageCmd;
  delete pBuildDensityImageCmd;
  delete pBuildMassImageCmd;
  delete pDoNotBuildVoxelsCmd;
  delete pIsoCenterRotationFlagCmd;
  delete pSetMaxOutOfRangeFractionCmd;
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
void GateVImageVolumeMessenger::SetNewValue(G4UIcommand* command,
                                            G4String newValue)
{
  GateMessage("Volume",5,"GateVImageVolumeMessenger::SetNewValue "
              << command->GetCommandPath()
              << " newValue=" << newValue << Gateendl);

  if (command == pImageFileNameCmd || command == pImageFileNameCmdDeprecated) {
    pVImageVolume->SetImageFilename(newValue);
    if (command == pImageFileNameCmdDeprecated) G4cout << "### WARNING ### SetImage is obsolete and will be removed from the next release. Please use setImage\n";
  }
  else if (command == pLabelToMaterialFileNameCmd) {
    pVImageVolume->SetLabelToMaterialTableFilename(newValue);
  }
  else if (command == pHUToMaterialFileNameCmd || command == pHUToMaterialFileNameCmdDeprecated) {
    pVImageVolume->SetHUToMaterialTableFilename(newValue);
    if (command == pHUToMaterialFileNameCmdDeprecated) G4cout << "### WARNING ### SetHUToMaterialFile is obsolete and will be removed from the next release. Please use setHUToMaterialFile\n";
  }
  else if (command == pRangeMaterialFileNameCmd) {
    pVImageVolume->SetRangeMaterialTableFilename(newValue);
  }
  else if (command == pIsoCenterCmd) {
    pVImageVolume->SetIsoCenter(pIsoCenterCmd->GetNew3VectorValue(newValue));
  }
  else if (command == pBuildDistanceTransfoCmd) {
    pVImageVolume->SetBuildDistanceTransfoFilename(newValue);
  }
  else if (command == pBuildLabeledImageCmd) {
    pVImageVolume->SetLabeledImageFilename(newValue);
  }
  else if (command == pBuildDensityImageCmd) {
    pVImageVolume->SetDensityImageFilename(newValue);
  }
  else if (command == pBuildMassImageCmd) {
    pVImageVolume->SetMassImageFilename(newValue);
  }
  else if (command == pDoNotBuildVoxelsCmd) {
    pVImageVolume->EnableBoundingBoxOnly(pDoNotBuildVoxelsCmd->GetNewBoolValue(newValue));
  }
  else if (command == pIsoCenterRotationFlagCmd) {
    pVImageVolume->SetIsoCenterRotationFlag(pIsoCenterRotationFlagCmd->GetNewBoolValue(newValue));
  }
  else if ( command == pSetMaxOutOfRangeFractionCmd) {
    pVImageVolume->SetMaxOutOfRangeFraction(pSetMaxOutOfRangeFractionCmd->GetNewDoubleValue(newValue));
  }
  // It is necessary to call GateVolumeMessenger::SetNewValue if the command
  // is not recognized
  else {
    GateVolumeMessenger::SetNewValue(command,newValue);
  }
}
//---------------------------------------------------------------------------

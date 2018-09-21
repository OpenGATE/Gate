/*----------------------
 GATE version name: gate_v6

 Copyright (C): OpenGATE Collaboration

 This software is distributed under the terms
 of the GNU Lesser General  Public Licence (LGPL)
 See LICENSE.md for further details
 ----------------------*/

#include "GateConfiguration.h"
#ifdef GATE_USE_RTK

#include "GateFixedForcedDetectionActorMessenger.hh"

#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithAString.hh"

GateFixedForcedDetectionActorMessenger::GateFixedForcedDetectionActorMessenger(GateFixedForcedDetectionActor* sensor) :
    GateActorMessenger(sensor), pActor(sensor)
  {
  BuildCommands(baseName + sensor->GetObjectName());
  /* Remove commands that we do not use in the fixed forced detection actor */
  delete pSetOverWriteFilesFlagCmd;
  pSetOverWriteFilesFlagCmd = NULL;
  delete pSetFileNameCmd;
  pSetFileNameCmd = NULL;
  delete pSetResetDataAtEachRunFlagCmd;
  pSetResetDataAtEachRunFlagCmd = NULL;
  }

GateFixedForcedDetectionActorMessenger::~GateFixedForcedDetectionActorMessenger()
  {
  }

void GateFixedForcedDetectionActorMessenger::BuildCommands(G4String base)
  {
  G4String guidance;
  G4String bb;

  bb = base + "/setDetector";
  pSetDetectorCmd = new G4UIcmdWithAString(bb, this);
  guidance = "Set the name of the volume used for detector (must be a Box).";
  pSetDetectorCmd->SetGuidance(guidance);

  bb = base + "/setDetectorResolution";
  pSetDetectorResolCmd = new GateUIcmdWith2Vector(bb, this);
  guidance = "Set the resolution of the detector (2D).";
  pSetDetectorResolCmd->SetGuidance(guidance);

  bb = base + "/binningFactor";
  pSetBinningFactorCmd = new GateUIcmdWith2Vector(bb, this);
  guidance = "Set u and v binning factor for computing several rays per pixel.";
  pSetBinningFactorCmd->SetGuidance(guidance);

  bb = base + "/geometryFilename";
  pSetGeometryFilenameCmd = new G4UIcmdWithAString(bb, this);
  guidance = "Set the file name for the output RTK geometry filename corresponding to primary projections.";
  pSetGeometryFilenameCmd->SetGuidance(guidance);

  bb = base + "/primaryFilename";
  pSetPrimaryFilenameCmd = new G4UIcmdWithAString(bb, this);
  guidance = "Set the file name for the primary x-rays (printf format with runId as a single parameter).";
  pSetPrimaryFilenameCmd->SetGuidance(guidance);

  bb = base + "/materialMuFilename";
  pSetMaterialMuFilenameCmd = new G4UIcmdWithAString(bb, this);
  guidance = "Set the file name for the attenuation lookup table. Two paramaters: material index and energy.";
  pSetMaterialMuFilenameCmd->SetGuidance(guidance);

  bb = base + "/attenuationFilename";
  pSetAttenuationFilenameCmd = new G4UIcmdWithAString(bb, this);
  guidance = "Set the file name for the attenuation image (printf format with runId as a single parameter).";
  pSetAttenuationFilenameCmd->SetGuidance(guidance);

  bb = base + "/materialDeltaFilename";
  pSetMaterialDeltaFilenameCmd = new G4UIcmdWithAString(bb, this);
  guidance = "Set the output file name for the refractive index decrement lookup table. Two paramaters: material index and energy.";
  pSetMaterialDeltaFilenameCmd->SetGuidance(guidance);

  bb = base + "/fresnelFilename";
  pSetFresnelFilenameCmd = new G4UIcmdWithAString(bb, this);
  guidance = "Set the output file name for the Fresnel diffraction image (printf format with runId as a single parameter).";
  pSetFresnelFilenameCmd->SetGuidance(guidance);

  bb = base + "/responseDetectorFilename";
  pSetResponseDetectorFilenameCmd = new G4UIcmdWithAString(bb, this);
  guidance = G4String("Input response detector curve.");
  pSetResponseDetectorFilenameCmd->SetGuidance(guidance);

  bb = base + "/flatFieldFilename";
  pSetFlatFieldFilenameCmd = new G4UIcmdWithAString(bb, this);
  guidance = "Set the file name for the flat field image (printf format with runId as a single parameter).";
  pSetFlatFieldFilenameCmd->SetGuidance(guidance);

  bb = base + "/comptonFilename";
  pSetComptonFilenameCmd = new G4UIcmdWithAString(bb, this);
  guidance = "Set the file name for the Compton image (printf format with runId as a single parameter).";
  pSetComptonFilenameCmd->SetGuidance(guidance);

  bb = base + "/rayleighFilename";
  pSetRayleighFilenameCmd = new G4UIcmdWithAString(bb, this);
  guidance = "Set the file name for the Rayleigh image (printf format with runId as a single parameter).";
  pSetRayleighFilenameCmd->SetGuidance(guidance);

  bb = base + "/fluorescenceFilename";
  pSetFluorescenceFilenameCmd = new G4UIcmdWithAString(bb, this);
  guidance = "Set the file name for the fluorescence image (printf format with runId as a single parameter).";
  pSetFluorescenceFilenameCmd->SetGuidance(guidance);

  bb = base + "/isotropicPrimaryFilename";
  pSetIsotropicPrimaryFilenameCmd = new G4UIcmdWithAString(bb, this);
  guidance = "Set the file name for the isotropic primary image (printf format with runId as a single parameter).";
  pSetIsotropicPrimaryFilenameCmd->SetGuidance(guidance);

  bb = base + "/setSourceType";
  pSetSourceTypeCmd = new G4UIcmdWithAString(bb, this);
  guidance = "Set the type of source.";
  pSetSourceTypeCmd->SetGuidance(guidance);

  bb = base + "/generatePhotons";
  pSetGeneratePhotonsCmd = new G4UIcmdWithAString(bb, this);
  guidance = "Set the type of source.";
  pSetGeneratePhotonsCmd->SetGuidance(guidance);

  bb = base + "/connectARF";
  pSetARFCmd = new G4UIcmdWithAString(bb, this);
  guidance = "Connect ARF.";
  pSetARFCmd->SetGuidance(guidance);


  bb = base + "/secondaryFilename";
  pSetSecondaryFilenameCmd = new G4UIcmdWithAString(bb, this);
  guidance = "Set the file name for the scatter image (printf format with runId as a single parameter).";
  pSetSecondaryFilenameCmd->SetGuidance(guidance);

  bb = base + "/enableSquaredSecondary";
  pEnableSecondarySquaredCmd = new G4UIcmdWithABool(bb, this);
  guidance = G4String("Enable squared secondary computation");
  pEnableSecondarySquaredCmd->SetGuidance(guidance);

  bb = base + "/enableUncertaintySecondary";
  pEnableSecondaryUncertaintyCmd = new G4UIcmdWithABool(bb, this);
  guidance = G4String("Enable uncertainty secondary computation");
  pEnableSecondaryUncertaintyCmd->SetGuidance(guidance);

  bb = base + "/totalFilename";
  pSetTotalFilenameCmd = new G4UIcmdWithAString(bb, this);
  guidance = "Set the file name for the total (primary + scatter) image (printf format with runId as a single parameter).";
  pSetTotalFilenameCmd->SetGuidance(guidance);

  bb = base + "/phaseSpaceFilename";
  pSetPhaseSpaceFilenameCmd = new G4UIcmdWithAString(bb, this);
  guidance = "Set the file name for storing all interactions in a phase space file in root format.";
  pSetPhaseSpaceFilenameCmd->SetGuidance(guidance);

  bb = base + "/setInputRTKGeometryFilename";
  pSetInputRTKGeometryFilenameCmd = new G4UIcmdWithAString(bb, this);
  guidance = "Set filename for using an RTK geometry file as input geometry.";
  pSetInputRTKGeometryFilenameCmd->SetGuidance(guidance);

  bb = base + "/noisePrimaryNumber";
  pSetNoisePrimaryCmd = new G4UIcmdWithAnInteger(bb, this);
  guidance = "Set a number of primary for noise estimate in a phase space file in root format.";
  pSetNoisePrimaryCmd->SetGuidance(guidance);

  bb = base + "/energyResolvedBinSize";
  pEnergyResolvedBinSizeCmd = new G4UIcmdWithADoubleAndUnit(bb, this);
  guidance = "Set energy bin size for having an energy resolved output. Default is 0, i.e., off.";
  pEnergyResolvedBinSizeCmd->SetGuidance(guidance);

  }

void GateFixedForcedDetectionActorMessenger::SetNewValue(G4UIcommand* command, G4String param)
  {
  if (command == pSetDetectorCmd)
    {
    pActor->SetDetectorVolumeName(param);
    }
  if (command == pSetDetectorResolCmd)
    {
    pActor->SetDetectorResolution(pSetDetectorResolCmd->GetNew2VectorValue(param)[0],
                                  pSetDetectorResolCmd->GetNew2VectorValue(param)[1]);
    }
  if (command == pSetBinningFactorCmd)
    {
    pActor->SetBinningFactor(pSetBinningFactorCmd->GetNew2VectorValue(param)[0],
                                  pSetBinningFactorCmd->GetNew2VectorValue(param)[1]);
    }
  if (command == pSetGeometryFilenameCmd)
    {
    pActor->SetGeometryFilename(param);
    }
  if (command == pSetPrimaryFilenameCmd)
    {
    pActor->SetPrimaryFilename(param);
    }
  if (command == pSetMaterialMuFilenameCmd)
    {
    pActor->SetMaterialMuFilename(param);
    }
  if (command == pSetAttenuationFilenameCmd)
    {
    pActor->SetAttenuationFilename(param);
    }
  if (command == pSetMaterialDeltaFilenameCmd)
    {
    pActor->SetMaterialDeltaFilename(param);
    }
  if (command == pSetFresnelFilenameCmd)
    {
    pActor->SetFresnelFilename(param);
    }
  if (command == pSetFlatFieldFilenameCmd)
    {
    pActor->SetFlatFieldFilename(param);
    }
  if (command == pSetComptonFilenameCmd)
    {
    pActor->SetComptonFilename(param);
    }
  if (command == pSetRayleighFilenameCmd)
    {
    pActor->SetRayleighFilename(param);
    }
  if (command == pSetResponseDetectorFilenameCmd)
    {
    pActor->SetResponseDetectorFilename(param);
    }
  if (command == pSetFluorescenceFilenameCmd)
    {
    pActor->SetFluorescenceFilename(param);
    }
  if (command == pSetIsotropicPrimaryFilenameCmd)
    {
    pActor->SetIsotropicPrimaryFilename(param);
    }
  if (command == pSetSourceTypeCmd)
    {
    pActor->SetSourceType(param);
    }
  if (command == pSetGeneratePhotonsCmd)
    {
    pActor->SetGeneratePhotons(param);
    }
  if (command == pSetARFCmd)
     {
     pActor->SetARF(param);
     }

  if (command == pSetSecondaryFilenameCmd)
    {
    pActor->SetSecondaryFilename(param);
    }
  if (command == pEnableSecondarySquaredCmd)
    {
    pActor->EnableSecondarySquaredImage(pEnableSecondarySquaredCmd->GetNewBoolValue(param));
    }
  if (command == pEnableSecondaryUncertaintyCmd)
    {
    pActor->EnableSecondaryUncertaintyImage(pEnableSecondaryUncertaintyCmd->GetNewBoolValue(param));
    }
  if (command == pSetTotalFilenameCmd)
    {
    pActor->SetTotalFilename(param);
    }
  if (command == pSetPhaseSpaceFilenameCmd)
    {
    pActor->SetPhaseSpaceFilename(param);
    }
  if (command == pSetInputRTKGeometryFilenameCmd)
    {
    pActor->SetInputRTKGeometryFilename(param);
    }
  if (command == pSetNoisePrimaryCmd)
    {
    pActor->SetNoisePrimary(pSetNoisePrimaryCmd->GetNewIntValue(param));
    }
  if (command == pEnergyResolvedBinSizeCmd)
    {
    pActor->SetEnergyResolvedBinSize(pEnergyResolvedBinSizeCmd->GetNewDoubleValue(param));
    }

  GateActorMessenger::SetNewValue(command, param);
  }

#endif /* GATE_USE_RTK */

/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "GatePhaseSpaceActorMessenger.hh"

#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithoutParameter.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "GatePhaseSpaceActor.hh"

//-----------------------------------------------------------------------------
GatePhaseSpaceActorMessenger::GatePhaseSpaceActorMessenger(GatePhaseSpaceActor *sensor)
    : GateActorMessenger(sensor), pActor(sensor)
{
    BuildCommands(baseName + sensor->GetObjectName());
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
GatePhaseSpaceActorMessenger::~GatePhaseSpaceActorMessenger()
{
    delete pEnableChargeCmd;
    delete pEnableAtomicNumberCmd;
    delete pEnableElectronicDEDXCmd;
    delete pEnableTotalDEDXCmd;
    delete pEnableMassCmd;
    delete pEnableEkineCmd;
    delete pEnablePositionXCmd;
    delete pEnablePositionYCmd;
    delete pEnablePositionZCmd;
    delete pEnableDirectionXCmd;
    delete pEnableDirectionYCmd;
    delete pEnableDirectionZCmd;
    delete pEnableProdVolumeCmd;
    delete pEnableProdProcessCmd;
    delete pEnableParticleNameCmd;
    delete pCoordinateInVolumeFrameCmd;
    delete pEnableWeightCmd;
    delete pEnableTimeCmd;
    delete pEnableTimeFromBeginOfEventCmd;
    delete pMaxSizeCmd;
    delete pInOrOutGoingParticlesCmd;
    delete pEnableSecCmd;
    delete pEnableStoreAllStepCmd;
    delete bEnablePrimaryEnergyCmd;
    delete bCoordinateFrameCmd;
    delete bEnableLocalTimeCmd;
    delete bSpotIDFromSourceCmd;
    delete bEnableCompactCmd;
    delete bEnableEmissionPointCmd;
    delete bEnablePDGCodeCmd;
    delete pEnableNuclearFlagCmd;
    delete bEnableSphereProjection;
    delete bSetSphereProjectionCenter;
    delete bSetSphereProjectionRadius;
    delete bEnableTranslationAlongDirection;
    delete bSetTranslationAlongDirectionLength;
    delete pUseMaskCmd;
    delete pEnableKillCmd;
    delete pEnableTrackLengthCmd;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GatePhaseSpaceActorMessenger::BuildCommands(G4String base)
{
    G4String guidance;
    G4String bb;

    bb = base + "/enableCharge";
    pEnableChargeCmd = new G4UIcmdWithABool(bb, this);
    guidance = "Save electric charge of particles in the phase space file.";
    pEnableChargeCmd->SetGuidance(guidance);
    pEnableChargeCmd->SetParameterName("State", false);

    bb = base + "/enableAtomicNumber";
    pEnableAtomicNumberCmd = new G4UIcmdWithABool(bb, this);
    guidance = "Save atomic number of particles in the phase space file.";
    pEnableAtomicNumberCmd->SetGuidance(guidance);
    pEnableAtomicNumberCmd->SetParameterName("State", false);

    bb = base + "/enableElectronicDEDX";
    pEnableElectronicDEDXCmd = new G4UIcmdWithABool(bb, this);
    guidance = "Save electronic energy loss de/dx of particles in the phase space file.";
    pEnableElectronicDEDXCmd->SetGuidance(guidance);
    pEnableElectronicDEDXCmd->SetParameterName("State", false);

    bb = base + "/enableTotalDEDX";
    pEnableTotalDEDXCmd = new G4UIcmdWithABool(bb, this);
    guidance = "Save total energy loss de/dx  of particles in the phase space file.";
    pEnableTotalDEDXCmd->SetGuidance(guidance);
    pEnableTotalDEDXCmd->SetParameterName("State", false);

    bb = base + "/enableEkine";
    pEnableEkineCmd = new G4UIcmdWithABool(bb, this);
    guidance = "Save kinetic energy of particles in the phase space file.";
    pEnableEkineCmd->SetGuidance(guidance);
    pEnableEkineCmd->SetParameterName("State", false);

    bb = base + "/enableXPosition";
    pEnablePositionXCmd = new G4UIcmdWithABool(bb, this);
    guidance = "Save position of particles along X axis in the phase space file.";
    pEnablePositionXCmd->SetGuidance(guidance);
    pEnablePositionXCmd->SetParameterName("State", false);

    bb = base + "/enableXDirection";
    pEnableDirectionXCmd = new G4UIcmdWithABool(bb, this);
    guidance = "Save direction of particles along X axis in the phase space file.";
    pEnableDirectionXCmd->SetGuidance(guidance);
    pEnableDirectionXCmd->SetParameterName("State", false);

    bb = base + "/enableYPosition";
    pEnablePositionYCmd = new G4UIcmdWithABool(bb, this);
    guidance = "Save position of particles along Y axis in the phase space file.";
    pEnablePositionYCmd->SetGuidance(guidance);
    pEnablePositionYCmd->SetParameterName("State", false);

    bb = base + "/enableYDirection";
    pEnableDirectionYCmd = new G4UIcmdWithABool(bb, this);
    guidance = "Save direction of particles along Y axis in the phase space file.";
    pEnableDirectionYCmd->SetGuidance(guidance);
    pEnableDirectionYCmd->SetParameterName("State", false);

    bb = base + "/enableZPosition";
    pEnablePositionZCmd = new G4UIcmdWithABool(bb, this);
    guidance = "Save position of particles along Z axis in the phase space file.";
    pEnablePositionZCmd->SetGuidance(guidance);
    pEnablePositionZCmd->SetParameterName("State", false);

    bb = base + "/enableZDirection";
    pEnableDirectionZCmd = new G4UIcmdWithABool(bb, this);
    guidance = "Save direction of particles along Z axis in the phase space file.";
    pEnableDirectionZCmd->SetGuidance(guidance);
    pEnableDirectionZCmd->SetParameterName("State", false);

    bb = base + "/enableProductionVolume";
    pEnableProdVolumeCmd = new G4UIcmdWithABool(bb, this);
    guidance = "Save the name of the volume, where particle is created, in the phase space file.";
    pEnableProdVolumeCmd->SetGuidance(guidance);
    pEnableProdVolumeCmd->SetParameterName("State", false);

    bb = base + "/enableProductionProcess";
    pEnableProdProcessCmd = new G4UIcmdWithABool(bb, this);
    guidance = "Save the name of the process, which produced the particle, in the phase space file.";
    pEnableProdProcessCmd->SetGuidance(guidance);
    pEnableProdProcessCmd->SetParameterName("State", false);

    bb = base + "/enableParticleName";
    pEnableParticleNameCmd = new G4UIcmdWithABool(bb, this);
    guidance = "Save the name of particles in the phase space file.";
    pEnableParticleNameCmd->SetGuidance(guidance);
    pEnableParticleNameCmd->SetParameterName("State", false);

    bb = base + "/enableWeight";
    pEnableWeightCmd = new G4UIcmdWithABool(bb, this);
    guidance = "Save the weight of particles in the phase space file.";
    pEnableWeightCmd->SetGuidance(guidance);
    pEnableWeightCmd->SetParameterName("State", false);

    bb = base + "/enableTime";
    pEnableTimeCmd = new G4UIcmdWithABool(bb, this);
    guidance = "Save the time of particles in the phase space file.";
    pEnableTimeCmd->SetGuidance(guidance);
    pEnableTimeCmd->SetParameterName("State", false);

    bb = base+"/enableIonTime";
    pEnableIonTimeCmd = new G4UIcmdWithABool(bb,this);
    guidance = "Save the time of primary particles in the phase space file.";
    pEnableIonTimeCmd->SetGuidance(guidance);
    pEnableIonTimeCmd->SetParameterName("State",false);
    
    bb = base + "/enableTimeFromBeginOfEvent";
    pEnableTimeFromBeginOfEventCmd = new G4UIcmdWithABool(bb, this);
    guidance = "Save the time of particles since the begin of event time.";
    pEnableTimeFromBeginOfEventCmd->SetGuidance(guidance);
    pEnableTimeFromBeginOfEventCmd->SetParameterName("State", false);

    bb = base + "/enableTrackLength";
    pEnableTrackLengthCmd = new G4UIcmdWithABool(bb, this);
    guidance = "Save the track length in the phase space file.";
    pEnableTrackLengthCmd->SetGuidance(guidance);
    pEnableTrackLengthCmd->SetParameterName("State", false);

    bb = base + "/enableMass";
    pEnableMassCmd = new G4UIcmdWithABool(bb, this);
    guidance = "Save the mass of particles in the phase space file.";
    pEnableMassCmd->SetGuidance(guidance);
    pEnableMassCmd->SetParameterName("State", false);

    bb = base + "/storeSecondaries";
    pEnableSecCmd = new G4UIcmdWithABool(bb, this);
    guidance = "Store the secondary particles created in the attached volume.";
    pEnableSecCmd->SetGuidance(guidance);
    pEnableSecCmd->SetParameterName("State", false);

    bb = base + "/storeOutgoingParticles";
    pInOrOutGoingParticlesCmd = new G4UIcmdWithABool(bb, this);
    guidance = "Store the outgoing particles instead of incoming particles.";
    pInOrOutGoingParticlesCmd->SetGuidance(guidance);

    bb = base + "/storeAllStep";
    pEnableStoreAllStepCmd = new G4UIcmdWithABool(bb, this);
    guidance = "Store all steps inside the attached volume";
    pEnableStoreAllStepCmd->SetGuidance(guidance);
    pEnableStoreAllStepCmd->SetParameterName("State", false);

    bb = base + "/useVolumeFrame";
    pCoordinateInVolumeFrameCmd = new G4UIcmdWithABool(bb, this);
    guidance = "Record coordinate in the actor volume frame.";
    pCoordinateInVolumeFrameCmd->SetGuidance(guidance);

    bb = base + "/setMaxFileSize";
    pMaxSizeCmd = new G4UIcmdWithADoubleAndUnit(bb, this);
    guidance = G4String(
        "Set maximum size of the phase space file. When the file reaches the maximum size, a new file is automatically created.");
    pMaxSizeCmd->SetGuidance(guidance);
    pMaxSizeCmd->SetParameterName("Size", false);
    pMaxSizeCmd->SetUnitCategory("Memory size");

    bb = base + "/enablePrimaryEnergy";
    bEnablePrimaryEnergyCmd = new G4UIcmdWithABool(bb, this);
    guidance = "Store the energy of the primary particle for every hit.";
    bEnablePrimaryEnergyCmd->SetGuidance(guidance);

    bb = base + "/setCoordinateFrame";
    bCoordinateFrameCmd = new G4UIcmdWithAString(bb, this);
    guidance = "Store the hit coordinates in the frame of the frame passed as an argument.";
    bCoordinateFrameCmd->SetGuidance(guidance);
    bCoordinateFrameCmd->SetParameterName("Coordinate Frame", false);

    bb = base + "/enableLocalTime";
    bEnableLocalTimeCmd = new G4UIcmdWithABool(bb, this);
    guidance = "Store the local time.";
    bEnableLocalTimeCmd->SetGuidance(guidance);

    bb = base + "/enableSpotIDFromSource";
    bSpotIDFromSourceCmd = new G4UIcmdWithAString(bb, this);
    guidance = "Store the spotID of the primary particles from given source.";
    bSpotIDFromSourceCmd->SetGuidance(guidance);

    bb = base + "/enableCompact";
    bEnableCompactCmd = new G4UIcmdWithABool(bb, this);
    guidance = "Compact output by not storing trackID, runID, eventID, ProductionVolume, -track, -step and switching from ParticleType to PDGCode.";
    bEnableCompactCmd->SetGuidance(guidance);

    bb = base + "/enableEmissionPoint";
    bEnableEmissionPointCmd = new G4UIcmdWithABool(bb, this);
    guidance = "Store the emission point of each particle stored in the phasespace.";
    bEnableEmissionPointCmd->SetGuidance(guidance);

    bb = base + "/enablePDGCode";
    bEnablePDGCodeCmd = new G4UIcmdWithABool(bb, this);
    guidance = "Output the PDGCode instead of the ParticleName.";
    bEnablePDGCodeCmd->SetGuidance(guidance);

    bb = base + "/enableNuclearFlag";
    pEnableNuclearFlagCmd = new G4UIcmdWithABool(bb, this);
    guidance = "Save nuclear flags of particles in the phase space file.";
    pEnableNuclearFlagCmd->SetGuidance(guidance);
    pEnableNuclearFlagCmd->SetParameterName("State", false);

    bb = base + "/enableSphereProjection";
    bEnableSphereProjection = new G4UIcmdWithABool(bb, this);
    guidance = "Change the particle position point: project it on a sphere";
    bEnableSphereProjection->SetGuidance(guidance);

    bb = base + "/setSphereProjectionCenter";
    bSetSphereProjectionCenter = new G4UIcmdWith3VectorAndUnit(bb, this);
    guidance = "Set the center of the sphere where the points are projected";
    bSetSphereProjectionCenter->SetGuidance(guidance);

    bb = base + "/setSphereProjectionRadius";
    bSetSphereProjectionRadius = new G4UIcmdWithADoubleAndUnit(bb, this);
    guidance = "Set the radius of the sphere where the points are projected";
    bSetSphereProjectionRadius->SetGuidance(guidance);

    bb = base + "/enableTranslationAlongDirection";
    bEnableTranslationAlongDirection = new G4UIcmdWithABool(bb, this);
    guidance = "Change the particle position point: translate along the direction";
    bEnableTranslationAlongDirection->SetGuidance(guidance);

    bb = base + "/setTranslationAlongDirectionLength";
    bSetTranslationAlongDirectionLength = new G4UIcmdWithADoubleAndUnit(bb, this);
    guidance = "Set the translation length";
    bSetTranslationAlongDirectionLength->SetGuidance(guidance);

    bb = base + "/enableTOut";
    pEnableTOutCmd = new G4UIcmdWithABool(bb, this);
    guidance = "Store the time taken from the production to the leaving of the volume. Usefull only for the outgoing particles";
    pEnableTOutCmd->SetGuidance(guidance);
    pEnableTOutCmd->SetParameterName("State", false);

    bb = base + "/enableTProd";
    pEnableTProdCmd = new G4UIcmdWithABool(bb, this);
    guidance = "Save the production time of the particle wrt to the primary production (defined as a GlobalTime - LocalTime)";
    pEnableTProdCmd->SetGuidance(guidance);
    pEnableTProdCmd->SetParameterName("State", false);

    bb = base + "/useMask";
    pUseMaskCmd = new G4UIcmdWithAString(bb, this);
    guidance = "Store only if particle position is in mask (pixel value different from 0).";
    pUseMaskCmd->SetGuidance(guidance);

    bb = base + "/killParticle";
    pEnableKillCmd = new G4UIcmdWithABool(bb, this);
    guidance = "Kill particle once stored.";
    pEnableKillCmd->SetGuidance(guidance);
    pEnableKillCmd->SetParameterName("State", false);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GatePhaseSpaceActorMessenger::SetNewValue(G4UIcommand *command, G4String param)
{
    if (command == pEnableChargeCmd)
        pActor->SetIsChargeEnabled(pEnableChargeCmd->GetNewBoolValue(param));
    if (command == pEnableAtomicNumberCmd)
        pActor->SetIsAtomicNumberEnabled(pEnableAtomicNumberCmd->GetNewBoolValue(param));
    if (command == pEnableElectronicDEDXCmd)
        pActor->SetIsElectronicDEDXEnabled(pEnableElectronicDEDXCmd->GetNewBoolValue(param));
    if (command == pEnableTotalDEDXCmd)
        pActor->SetIsTotalDEDXEnabled(pEnableTotalDEDXCmd->GetNewBoolValue(param));
    if (command == pEnableEkineCmd)
        pActor->SetIsEkineEnabled(pEnableEkineCmd->GetNewBoolValue(param));
    if (command == pEnablePositionXCmd)
        pActor->SetIsXPositionEnabled(pEnablePositionXCmd->GetNewBoolValue(param));
    if (command == pEnableDirectionXCmd)
        pActor->SetIsXDirectionEnabled(pEnableDirectionXCmd->GetNewBoolValue(param));
    if (command == pEnablePositionYCmd)
        pActor->SetIsYPositionEnabled(pEnablePositionYCmd->GetNewBoolValue(param));
    if (command == pEnableDirectionYCmd)
        pActor->SetIsYDirectionEnabled(pEnableDirectionYCmd->GetNewBoolValue(param));
    if (command == pEnablePositionZCmd)
        pActor->SetIsZPositionEnabled(pEnablePositionZCmd->GetNewBoolValue(param));
    if (command == pEnableDirectionZCmd)
        pActor->SetIsZDirectionEnabled(pEnableDirectionZCmd->GetNewBoolValue(param));
    if (command == pEnableProdProcessCmd)
        pActor->SetIsProdProcessEnabled(pEnableProdProcessCmd->GetNewBoolValue(param));
    if (command == pEnableProdVolumeCmd)
        pActor->SetIsProdVolumeEnabled(pEnableProdVolumeCmd->GetNewBoolValue(param));
    if (command == pEnableParticleNameCmd)
        pActor->SetIsParticleNameEnabled(pEnableParticleNameCmd->GetNewBoolValue(param));
    if (command == pEnableWeightCmd)
        pActor->SetIsWeightEnabled(pEnableWeightCmd->GetNewBoolValue(param));
    if (command == pEnableTimeCmd)
        pActor->SetIsTimeEnabled(pEnableTimeCmd->GetNewBoolValue(param));
    if (command == pEnableIonTimeCmd)
        pActor->SetIsIonTimeEnabled(pEnableIonTimeCmd->GetNewBoolValue(param));
    if (command == pEnableTimeFromBeginOfEventCmd)
        pActor->SetIsTimeFromBeginOfEventEnabled(pEnableTimeFromBeginOfEventCmd->GetNewBoolValue(param));
    if (command == pEnableTrackLengthCmd)
        pActor->SetTrackLengthEnabled(pEnableTrackLengthCmd->GetNewBoolValue(param));
    if (command == pEnableMassCmd)
        pActor->SetIsMassEnabled(pEnableMassCmd->GetNewBoolValue(param));
    if (command == pCoordinateInVolumeFrameCmd)
        pActor->SetUseVolumeFrame(pCoordinateInVolumeFrameCmd->GetNewBoolValue(param));
    if (command == pInOrOutGoingParticlesCmd)
        pActor->SetStoreOutgoingParticles(pInOrOutGoingParticlesCmd->GetNewBoolValue(param));
    if (command == pEnableStoreAllStepCmd)
        pActor->SetIsAllStep(pEnableStoreAllStepCmd->GetNewBoolValue(param));
    if (command == pEnableSecCmd)
        pActor->SetIsSecStored(pEnableSecCmd->GetNewBoolValue(param));
    if (command == pSaveEveryNEventsCmd || command == pSaveEveryNSecondsCmd)
        GateError(
            "saveEveryNEvents and saveEveryNSeconds commands are not available with phase space actor. But you can use the setMaxFileSize command.");
    if (command == pMaxSizeCmd)
        pActor->SetMaxFileSize(pMaxSizeCmd->GetNewDoubleValue(param));
    if (command == bEnablePrimaryEnergyCmd)
        pActor->SetIsPrimaryEnergyEnabled(bEnablePrimaryEnergyCmd->GetNewBoolValue(param));
    if (command == bEnableEmissionPointCmd)
        pActor->SetIsEmissionPointEnabled(bEnableEmissionPointCmd->GetNewBoolValue(param));
    if (command == bCoordinateFrameCmd)
    {
        pActor->SetCoordFrame(param);
        pActor->SetEnableCoordFrame();
    };
    if (command == bEnableLocalTimeCmd)
        pActor->SetIsLocalTimeEnabled(bEnableLocalTimeCmd->GetNewBoolValue(param));
    if (command == bSpotIDFromSourceCmd)
    {
        pActor->SetSpotIDFromSource(param);
        pActor->SetIsSpotIDEnabled();
    };
    if (command == bEnablePDGCodeCmd)
        pActor->SetEnablePDGCode(bEnablePDGCodeCmd->GetNewBoolValue(param));
    if (command == bEnableCompactCmd)
        pActor->SetEnabledCompact(bEnableCompactCmd->GetNewBoolValue(param));
    if (command == pEnableNuclearFlagCmd)
        pActor->SetIsNuclearFlagEnabled(pEnableNuclearFlagCmd->GetNewBoolValue(param));
    if (command == bEnableSphereProjection)
        pActor->SetEnabledSphereProjection(bEnableSphereProjection->GetNewBoolValue(param));
    if (command == bSetSphereProjectionCenter)
        pActor->SetSphereProjectionCenter(bSetSphereProjectionCenter->GetNew3VectorValue(param));
    if (command == bSetSphereProjectionRadius)
        pActor->SetSphereProjectionRadius(bSetSphereProjectionRadius->GetNewDoubleValue(param));

    if (command == bEnableTranslationAlongDirection)
        pActor->SetEnabledTranslationAlongDirection(bEnableTranslationAlongDirection->GetNewBoolValue(param));
    if (command == bSetTranslationAlongDirectionLength)
        pActor->SetTranslationAlongDirectionLength(bSetTranslationAlongDirectionLength->GetNewDoubleValue(param));

    if (command == pEnableTOutCmd)
        pActor->SetIsTOutEnabled(pEnableTOutCmd->GetNewBoolValue(param));
    if(command == pEnableTProdCmd)
        pActor->SetIsTProdEnabled(pEnableTProdCmd->GetNewBoolValue(param));
    if (command == pEnableTProdCmd)
        pActor->SetIsTProdEnabled(pEnableTProdCmd->GetNewBoolValue(param));
    if (command == pUseMaskCmd)
        pActor->SetMaskFilename(param);
    if (command == pEnableKillCmd)
        pActor->SetKillParticleFlag(pEnableKillCmd->GetNewBoolValue(param));

    GateActorMessenger::SetNewValue(command, param);
}
//-----------------------------------------------------------------------------

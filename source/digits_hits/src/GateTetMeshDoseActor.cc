/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/
#include <fstream>
#include <vector>
#include <map>
#include <algorithm>
#include <cmath>
#include <limits>

#include <G4String.hh>
#include <G4Types.hh>
#include <G4Run.hh>
#include <G4VHitsCollection.hh>
#include <G4THitsMap.hh>
#include <G4Tet.hh>
#include <G4LogicalVolume.hh>
#include <G4AssemblyVolume.hh>

#include "GateMessageManager.hh"
#include "GateVVolume.hh"
#include "GateTetMeshBox.hh"
#include "GateVActor.hh"
#include "GateActorMessenger.hh"

#include "GateTetMeshDoseActor.hh"


//----------------------------------------------------------------------------------------


GateTetMeshDoseActor::GateTetMeshDoseActor(G4String name, G4int depth)
  : GateVActor(name, depth), mEvtDoseMap(), mRunCounter(),
    mRunData(), pMessenger(new GateActorMessenger(this))
{
}

void GateTetMeshDoseActor::Construct()
{
  GateMessage("Actor", 1, "Constructing TetMeshDoseActor '" << 
                          GateNamedObject::GetObjectName() << "'." << Gateendl);
  
  // enable callbacks which are implemented
  EnableBeginOfRunAction(true);
  EnableEndOfRunAction(true);
  EnableEndOfEventAction(true);
  EnableUserSteppingAction(true);

  mRunCounter = 0;
}

//----------------------------------------------------------------------------------------

void GateTetMeshDoseActor::BeginOfRunAction(const G4Run* run)
{
  // before the first run
  if (mRunCounter == 0)
  {
    // check whether the volume is in fact a tetrahedral mesh
    GateTetMeshBox* tetMeshBox = dynamic_cast<GateTetMeshBox*>(GateVActor::mVolume);
    if (tetMeshBox == nullptr)
    {
      GateError("Actor '" << GateNamedObject::GetObjectName() << "' is attached" <<
                " to volume of incorrect type. Please attach to TetMeshBox.");
    }

    // only now init data
    InitData();
  }
    
  // In general: the default BeginOfRunAction
  GateVActor::BeginOfRunAction(run);
}

void GateTetMeshDoseActor::EndOfRunAction(const G4Run* run)
{
  // number of events processed in this run, i.e. number of "histories"
  G4int n = run->GetNumberOfEvent();
  
  // Calculate relative uncertainty of dose values
  for (Estimators& tetEstimators : mRunData)
  {
    G4double mean = tetEstimators.dose;
    G4double squared = tetEstimators.sumOfSquaredDose;
    
    if (n > 1 && mean > 0)
    {
      // history by history method
      tetEstimators.relativeUncertainty =
        std::sqrt( (1.0 / (n - 1)) * (squared  / n - (mean / n) * (mean / n)) ) * n / mean;
    }
    else
    {
      // if N in {0, 1} or mean = 0, the relative uncertainty is ill-defined
      tetEstimators.relativeUncertainty = std::numeric_limits<G4double>::infinity();
    }
  }

  SaveData();
  ++mRunCounter;
}

void GateTetMeshDoseActor::EndOfEventAction(const G4Event*)
{  
  // get the TetMeshBox this actor is attached to
  GateTetMeshBox* tetMeshBox = dynamic_cast<GateTetMeshBox*>(GateVActor::mVolume);
  
  // Accumulate event dose in the run's dose map.
  for (const auto& keyValuePair : mEvtDoseMap)
  {
    G4int iTetrahedron = tetMeshBox->GetTetIndex(keyValuePair.first);
    G4double dose = keyValuePair.second;

    Estimators& tetEstimator = mRunData[iTetrahedron];
    tetEstimator.dose += dose;
    tetEstimator.sumOfSquaredDose += dose * dose;
  }
}

void GateTetMeshDoseActor::InitData()
{
  // get the TetMeshBox this actor is attached to
  GateTetMeshBox* tetMeshBox = dynamic_cast<GateTetMeshBox*>(GateVActor::mVolume);

  std::size_t nTetrahedra = tetMeshBox->GetNumberOfTetrahedra();
  Estimators initialEstimates{0.0, 0.0, std::numeric_limits<G4double>::infinity()};
  
  mRunData.clear();
  mRunData.resize(nTetrahedra, initialEstimates);
}

void GateTetMeshDoseActor::SaveData()
{
  // get the TetMeshBox this actor is attached to
  GateTetMeshBox* tetMeshBox = dynamic_cast<GateTetMeshBox*>(GateVActor::mVolume);
  
  std::ofstream csvTable(GateVActor::GetSaveFilename(), std::ofstream::out);

  // header of csv file
  csvTable << "# Tetrahedron-ID, Dose [Gy], Relative Uncertainty, "
           << "Sum of Squared Dose [Gy^2], Volume [cm^3], "
           << "Density [g / cm^3], Region Marker" << std::endl;

  for (std::size_t iTet = 0; iTet < tetMeshBox->GetNumberOfTetrahedra(); ++iTet)
  {
    const G4LogicalVolume* tetLogical = tetMeshBox->GetTetLogical(iTet);

    G4double dose = mRunData[iTet].dose;
    G4double relativeUncertainty = mRunData[iTet].relativeUncertainty;
    G4double sumOfSquaredDose = mRunData[iTet].sumOfSquaredDose;
    G4double cubicVolume = tetLogical->GetSolid()->GetCubicVolume();
    G4double density = tetLogical->GetMaterial()->GetDensity();
    G4int regionMarker = tetMeshBox->GetRegionMarker(iTet);

    csvTable << iTet << ", " << dose / gray << ", " << relativeUncertainty << ", "
             << sumOfSquaredDose / (gray*gray) << ", " << cubicVolume / (cm*cm*cm) << ", "
             << density * (cm*cm*cm) / g << ", " << regionMarker << std::endl;
  }

  GateMessage("Actor", 1, GateNamedObject::GetObjectName() << ": Saving dose to file '" << 
                          GateVActor::GetSaveFilename() << "'." << Gateendl);
  csvTable.close();
}

void GateTetMeshDoseActor::ResetData()
{
  Estimators initialEstimates{0.0, 0.0, std::numeric_limits<G4double>::infinity()};
  std::fill(mRunData.begin(), mRunData.end(), initialEstimates);
}

//----------------------------------------------------------------------------------------

void GateTetMeshDoseActor::Initialize(G4HCofThisEvent*)
{
  mEvtDoseMap.clear();
}

void GateTetMeshDoseActor::EndOfEvent(G4HCofThisEvent*)
{
}

void GateTetMeshDoseActor::clear()
{
  mEvtDoseMap.clear();
}

// compare with G4PSDoseScorer
void GateTetMeshDoseActor::UserSteppingAction(const GateVVolume*, const G4Step* aStep)
{
  G4VPhysicalVolume* physVol = aStep->GetPreStepPoint()->GetPhysicalVolume();
  G4int copyNum = physVol->GetCopyNo();

  G4double edep = aStep->GetTotalEnergyDeposit();
  G4double weight = aStep->GetPreStepPoint()->GetWeight();

  // discard steps in bounding box volume or without energy deposition
  if (copyNum == 0 || edep == 0)
    return;

  G4LogicalVolume* logVol = physVol->GetLogicalVolume();
  G4double cubicVolume = logVol->GetSolid()->GetCubicVolume();
  G4double density = logVol->GetMaterial()->GetDensity();

  G4double dose = (edep * weight) / (density * cubicVolume);

  // accumulate or add
  if (mEvtDoseMap.find(copyNum) == mEvtDoseMap.end())
  {
    mEvtDoseMap[copyNum] = dose;
  }
  else
  {
    mEvtDoseMap[copyNum] += dose;    
  }
}

/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


/*
  \brief Class G4XrayBoundaryProcess :
  \brief
*/

#include "G4XrayBoundaryProcess.hh"


/////////////////////////
// Class Implementation
/////////////////////////

//////////////
// Operators
//////////////

/////////////////
// Constructors
/////////////////
G4XrayBoundaryProcess::G4XrayBoundaryProcess(const G4String &processName, G4ProcessType type) : G4VDiscreteProcess(processName, type) {
    if (verboseLevel > 0) {
        G4cout << GetProcessName() << " is created " << G4endl;
    }

    Material1 = NULL;
    Material2 = NULL;

    kCarTolerance = G4GeometryTolerance::GetInstance()
            ->GetSurfaceTolerance();

    TotalMomentum = 0.;
    Rindex1 = Rindex2 = 1.;
    cost1 = cost2 = sint1 = sint2 = 0.;

}

////////////////
// Destructors
////////////////
G4XrayBoundaryProcess::~G4XrayBoundaryProcess() {}

////////////
// Methods
////////////

// PostStepDoIt
// ------------
//
G4VParticleChange *G4XrayBoundaryProcess::PostStepDoIt(const G4Track &aTrack, const G4Step &aStep) {
    // Get hyperStep from  G4ParallelWorldProcess
    //  NOTE: PostSetpDoIt of this process should be
    //        invoked after G4ParallelWorldProcess!

    aParticleChange.Initialize(aTrack);
    aParticleChange.ProposeVelocity(aTrack.GetVelocity());

    const G4Step *pStep = &aStep;

    const G4Step *hStep = G4ParallelWorldProcess::GetHyperStep();

    if (hStep) pStep = hStep;

    G4bool isOnBoundary =
            (pStep->GetPostStepPoint()->GetStepStatus() == fGeomBoundary);

    if (isOnBoundary) {
        Material1 = pStep->GetPreStepPoint()->GetMaterial();
        Material2 = pStep->GetPostStepPoint()->GetMaterial();
    } else {
        return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
    }

    G4VPhysicalVolume *thePrePV  =
            pStep->GetPreStepPoint() ->GetPhysicalVolume();
    G4VPhysicalVolume *thePostPV =
            pStep->GetPostStepPoint()->GetPhysicalVolume();

    if ( verboseLevel > 0 ) {
        G4cout << " X-ray at Boundary! " << G4endl;
        if (thePrePV)  G4cout << " thePrePV:  " << thePrePV->GetName()  << G4endl;
        if (thePostPV) G4cout << " thePostPV: " << thePostPV->GetName() << G4endl;
    }

    if (aTrack.GetStepLength() <= kCarTolerance / 2) {
        return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
    }

    const G4DynamicParticle *aParticle = aTrack.GetDynamicParticle();

    TotalMomentum     = aParticle->GetTotalMomentum();
    OldMomentum       = aParticle->GetMomentumDirection();
    OldPolarization   = aParticle->GetPolarization();

    if ( verboseLevel > 0 ) {
        G4cout << " Old Momentum Direction: " << OldMomentum     << G4endl;
        G4cout << " Old Polarization:       " << OldPolarization << G4endl;
    }

    G4ThreeVector theGlobalPoint = pStep->GetPostStepPoint()->GetPosition();

    G4bool valid;
    //  Use the new method for Exit Normal in global coordinates,
    //    which provides the normal more reliably.

    // ID of Navigator which limits step

    G4int hNavId = G4ParallelWorldProcess::GetHypNavigatorID();
    std::vector<G4Navigator *>::iterator iNav =
            G4TransportationManager::GetTransportationManager()->
            GetActiveNavigatorsIterator();
    theGlobalNormal =
            (iNav[hNavId])->GetGlobalExitNormal(theGlobalPoint, &valid);

    if (valid) {
        theGlobalNormal = -theGlobalNormal;
    } else {
        G4ExceptionDescription ed;
        ed << " G4XrayBoundaryProcess/PostStepDoIt(): "
           << " The Navigator reports that it returned an invalid normal"
           << G4endl;
        G4Exception("G4XrayBoundaryProcess::PostStepDoIt", "XrayBoun01",
                    EventMustBeAborted, ed,
                    "Invalid Surface Normal - Geometry must return valid surface normal");
    }

    Rindex1 = GetRindex(Material1, TotalMomentum);
    Rindex2 = GetRindex(Material2, TotalMomentum);

    G4double PdotN = OldMomentum * theGlobalNormal;
    cost1 = - PdotN;

    if (std::abs(cost1) < 1.0 - kCarTolerance) {
        sint1 = std::sqrt(1. - cost1 * cost1);
        sint2 = sint1 * Rindex1 / Rindex2; // *** Snell's Law ***
    } else {
        sint1 = 0.0;
        sint2 = 0.0;
    }

    if (sint2 >= 1.0) {
        DoReflection(); // *** Total reflection ***
    } else {
        if (cost1 > 0.0) {
            cost2 =  std::sqrt(1. - sint2 * sint2);
        } else {
            cost2 = -std::sqrt(1. - sint2 * sint2);
        }

        if (sint1 > 0.0) {      // incident ray oblique
	  // Compute the reflectance and sample a uniform random to test if reflected or transmitted
	  G4double diti = (Rindex1*cost1 - Rindex2*cost2) / (Rindex1*cost1 + Rindex2*cost2);
	  G4double ditj = (Rindex1*cost2 - Rindex2*cost1) / (Rindex1*cost2 + Rindex2*cost1);
	  G4double reflectance = 0.5 * ( diti*diti + ditj*ditj ) ;
	  G4double isReflected = CLHEP::RandFlat::shoot() ;
	  if (isReflected < reflectance) { // if reflectance test passed
	    DoReflection(); // *** Total reflection ***
	  } else {
            G4double alpha = cost1 - cost2 * (Rindex2 / Rindex1);
            NewMomentum = OldMomentum + alpha * theGlobalNormal;
	  }
        } else {                // incident ray perpendicular ==> transmission
	  NewMomentum = OldMomentum;
	  NewPolarization = OldPolarization;
        }
    }

    NewMomentum = NewMomentum.unit();
    NewPolarization = NewPolarization.unit();

    if ( verboseLevel > 0 ) {
        G4cout << " New Momentum Direction: " << NewMomentum     << G4endl;
        G4cout << " New Polarization:       " << NewPolarization << G4endl;
    }

    aParticleChange.ProposePolarization(OldPolarization);
    aParticleChange.ProposeMomentumDirection(NewMomentum);

    return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
}

// GetMeanFreePath
// ---------------
//
G4double G4XrayBoundaryProcess::GetMeanFreePath(const G4Track &, G4double , G4ForceCondition *condition) {
    *condition = Forced;

    return DBL_MAX;
}

G4double G4XrayBoundaryProcess::GetIncidentAngle() {
    G4double PdotN = OldMomentum * theGlobalNormal;
    G4double magP = OldMomentum.mag();
    G4double magN = theGlobalNormal.mag();
    G4double incidentangle = pi - std::acos(PdotN / (magP * magN));

    return incidentangle;
}

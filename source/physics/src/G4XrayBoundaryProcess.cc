//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
////////////////////////////////////////////////////////////////////////
// X-ray Boundary Process Class Implementation
////////////////////////////////////////////////////////////////////////
//
// File:        G4XrayBoundaryProcess.cc
// Description: Discrete Process -- reflection/refraction at
//                                  interfaces
// Version:     1.0
// Created:     2016-05-26
// Modified:
//
// Author:      Zhenjie Cen
// mail:        zhenjie.cen@creatis.insa-lyon.fr
//
////////////////////////////////////////////////////////////////////////

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
    //SetProcessSubType(fXrayBoundary);

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

    // What is this used for?
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
        ed << " G4OpBoundaryProcess/PostStepDoIt(): "
           << " The Navigator reports that it returned an invalid normal"
           << G4endl;
        G4Exception("G4OpBoundaryProcess::PostStepDoIt", "OpBoun01",
                    EventMustBeAborted, ed,
                    "Invalid Surface Normal - Geometry must return valid surface normal");
    }


//    G4MaterialPropertiesTable *aMaterialPropertiesTable;
//    G4MaterialPropertyVector *Rindex;

//    aMaterialPropertiesTable = Material1->GetMaterialPropertiesTable();

//    if (aMaterialPropertiesTable) {
//        Rindex = aMaterialPropertiesTable->GetProperty("RINDEX");
//    } else {
//        return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
//    }

//    if (Rindex) {
//        Rindex1 = Rindex->Value(TotalMomentum);
//    } else {
//        return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
//    }

//    aMaterialPropertiesTable = Material2->GetMaterialPropertiesTable();

//    if (aMaterialPropertiesTable) {
//        Rindex = aMaterialPropertiesTable->GetProperty("RINDEX");
//    } else {
//        return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
//    }

//    if (Rindex) {
//        Rindex2 = Rindex->Value(TotalMomentum);
//    } else {
//        return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
//    }

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
    double Refractive_Index_Re(const char compound[], double E, double density);
    if (sint2 >= 1.0) {
        DoReflection(); // *** Total reflection ***
    } else {
        if (cost1 > 0.0) {
            cost2 =  std::sqrt(1. - sint2 * sint2);
        } else {
            cost2 = -std::sqrt(1. - sint2 * sint2);
        }

        if (sint1 > 0.0) {      // incident ray oblique
            G4double alpha = cost1 - cost2 * (Rindex2 / Rindex1);
            NewMomentum = OldMomentum + alpha * theGlobalNormal;
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

//    G4cout << "Incident angle: " << GetIncidentAngle() * 180 / pi << " refraction angle: " << std::asin(sint2) * 180 / pi << G4endl;

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

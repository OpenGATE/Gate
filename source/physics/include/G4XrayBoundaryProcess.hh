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
//
// $Id: G4XrayBoundaryProcess.hh 84717 2014-10-20 07:39:47Z gcosmo $
//
//
////////////////////////////////////////////////////////////////////////
// X-ray Boundary Process Class Definition
////////////////////////////////////////////////////////////////////////
//
// File:        G4XrayBoundaryProcess.hh
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

#ifndef G4XrayBoundaryProcess_h
#define G4XrayBoundaryProcess_h 1

/////////////
// Includes
/////////////

#include "G4Step.hh"
#include "G4VDiscreteProcess.hh"
#include "G4EmProcessSubType.hh"
#include "G4PhysicalConstants.hh"

#include "G4ParallelWorldProcess.hh"
#include "G4Gamma.hh"
#include "G4TransportationManager.hh"
#include "G4GeometryTolerance.hh"

// Class Description:
// Discrete Process -- reflection/refraction at interfaces.
// Class inherits publicly from G4VDiscreteProcess.
// Class Description - End:

/////////////////////
// Class Definition
/////////////////////

class G4XrayBoundaryProcess : public G4VDiscreteProcess {

public:

    ////////////////////////////////
    // Constructors and Destructor
    ////////////////////////////////

    G4XrayBoundaryProcess(const G4String &processName = "XrayBoundary", G4ProcessType type = fElectromagnetic);
    ~G4XrayBoundaryProcess();

private:

    G4XrayBoundaryProcess(const G4XrayBoundaryProcess &right);

    //////////////
    // Operators
    //////////////

    G4XrayBoundaryProcess &operator=(const G4XrayBoundaryProcess &right);

public:

    ////////////
    // Methods
    ////////////

    G4bool IsApplicable(const G4ParticleDefinition &aParticleType);

    void DoReflection();

    G4double GetMeanFreePath(const G4Track &aTrack, G4double , G4ForceCondition *condition);
    // Returns infinity; i. e. the process does not limit the step,
    // but sets the 'Forced' condition for the DoIt to be invoked at
    // every step. However, only at a boundary will any action be
    // taken.

    G4VParticleChange *PostStepDoIt(const G4Track &aTrack, const G4Step &aStep);
    // This is the method implementing boundary processes.

private:

    G4double GetIncidentAngle();

private:
    G4double TotalMomentum;
    G4ThreeVector OldMomentum;
    G4ThreeVector OldPolarization;

    G4ThreeVector NewMomentum;
    G4ThreeVector NewPolarization;

    G4ThreeVector theGlobalNormal;

    G4double Rindex1;
    G4double Rindex2;

    G4double cost1, cost2, sint1, sint2;

    G4Material *Material1;
    G4Material *Material2;

    G4double kCarTolerance;
};

////////////////////
// Inline methods
////////////////////

inline
G4bool G4XrayBoundaryProcess::IsApplicable(const G4ParticleDefinition &aParticleType) {
    return ( &aParticleType == G4Gamma::Gamma() );
}

inline
void G4XrayBoundaryProcess::DoReflection() {
    G4double PdotN = OldMomentum * theGlobalNormal;
    NewMomentum = OldMomentum - (2.*PdotN) * theGlobalNormal;
}

#endif /* G4XrayBoundaryProcess_h */

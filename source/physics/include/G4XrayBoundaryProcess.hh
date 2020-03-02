/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*!
  \class  G4XrayBoundaryProcess
  \author zhenjie.cen@creatis.insa-lyon.fr
  \ref    WANG, Zhentian, HUANG, Zhifeng, ZHANG, Li, et al. Implement X-ray refraction effect in Geant4 for phase contrast imaging.
          In : 2009 IEEE Nuclear Science Symposium Conference Record (NSS/MIC). IEEE, 2009. p. 2395-2398.
*/

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
#include "G4SystemOfUnits.hh"

#include "GateConfiguration.h"

#ifdef GATE_USE_XRAYLIB
#include <xraylib.h>
#endif

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

    G4double GetRindex(G4Material *Material, G4double Energy);

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

inline
#ifdef GATE_USE_XRAYLIB
G4double G4XrayBoundaryProcess::GetRindex(G4Material *Material, G4double Energy) {
#else
G4double G4XrayBoundaryProcess::GetRindex(G4Material *Material, G4double) {
#endif
    G4double delta = 0.0;
    G4double Density = Material->GetDensity() / (g / cm3);

#ifdef GATE_USE_XRAYLIB
#if XRAYLIB_MAJOR > 3
    xrl_error *error = NULL;
    for (unsigned int i = 0; i < Material->GetElementVector()->size(); ++i) {
      delta += (1 - Refractive_Index_Re(Material->GetElementVector()->at(i)->GetSymbol(), Energy/(keV), 1.0,&error)) * Material->GetFractionVector()[i];
      if (error != NULL) {
	G4cerr << "error message: " << error->message << "\n";
	xrl_clear_error(&error);
      }
    }
#else
    for (unsigned int i = 0; i < Material->GetElementVector()->size(); ++i)
      delta += (1 - Refractive_Index_Re(Material->GetElementVector()->at(i)->GetSymbol(), Energy/(keV), 1.0)) * Material->GetFractionVector()[i];
#endif    
#else
    G4Exception( "G4XrayBoundaryProcess::GetRindex", "GetRindex", FatalException, "Xraylib is not available\n");
#endif

    return 1 - delta * Density;
}

#endif /* G4XrayBoundaryProcess_h */

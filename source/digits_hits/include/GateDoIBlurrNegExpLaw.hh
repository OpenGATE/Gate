/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


/*!
  \class  GateDoIBlurrNegExpLaw

  The user can choose the direction in which he wants to applied the DoI model.

  Last modification (Adaptation to GND): May 2023 by Mohamed-Jordan Soumano mjsoumano@yahoo.com
*/


#ifndef GateDoIBlurrNegExpLaw_h
#define GateDoIBlurrNegExpLaw_h 1

#include "GateVDoILaw.hh"
#include "GateDoIBlurrNegExpLawMessenger.hh"

#include "GateTools.hh"
#include "G4UnitsTable.hh"
#include "G4VoxelLimits.hh"


#include "G4Transform3D.hh"
#include "G4VSolid.hh"

class GateDoIBlurrNegExpLaw  : public GateVDoILaw {

public :
    GateDoIBlurrNegExpLaw(const G4String& itsName);
    virtual ~GateDoIBlurrNegExpLaw() {delete m_messenger;}

     virtual void ComputeDoI(GateDigi* digi, G4ThreeVector axis);

    virtual void DescribeMyself (size_t ident=0) const;

    inline G4double GetExpInvDecayConst() const { return m_mu; }
    inline G4double GetEntranceFWHM() const { return m_entFWHM; }


    inline void SetExpInvDecayConst(G4double mu) { m_mu = mu; }
    inline void SetEntranceFWHM(G4double entFWHM) { m_entFWHM = entFWHM; }


private :

    G4double m_mu;
    G4double m_entFWHM;

    GateDoIBlurrNegExpLawMessenger* m_messenger;
    G4VoxelLimits limits;
    G4double DoImin, DoImax;
    G4AffineTransform at;



    G4ThreeVector xAxis;
    G4ThreeVector yAxis;
    G4ThreeVector zAxis;


};

#endif

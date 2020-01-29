


#ifndef GateDualLayerLaw_h
#define GateDualLayerLaw_h 1

#include "GateVDoILaw.hh"
#include "GateDualLayerLawMessenger.hh"

#include "GateTools.hh"
#include "G4UnitsTable.hh"
#include "G4VoxelLimits.hh"


#include "G4Transform3D.hh"
#include "G4VSolid.hh"

class GateDualLayerLaw  : public GateVDoILaw {

public :
    GateDualLayerLaw(const G4String& itsName);
    virtual ~GateDualLayerLaw() {delete m_messenger;}

     virtual void ComputeDoI(GatePulse *plse, G4ThreeVector axis);

    virtual void DescribeMyself (size_t ident=0) const;

private :

    GateDualLayerLawMessenger* m_messenger;
    G4VoxelLimits limits;
    G4double DoImin, DoImax;
    G4AffineTransform at;



    G4ThreeVector xAxis;
    G4ThreeVector yAxis;
    G4ThreeVector zAxis;


};

#endif

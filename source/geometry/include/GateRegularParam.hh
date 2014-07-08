/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GATEREGULARPARAMINSERTER_HH
#define GATEREGULARPARAMINSERTER_HH 1

#include "globals.hh"
#include "GateBox.hh"
#include "GateRegularParameterization.hh"
#include "G4ThreeVector.hh"

class GateRegularParameterized;
class G4PVParameterised;

class GateRegularParam : public GateBox
{
public:

    //! Constructor
    GateRegularParam(const G4String& itsName, GateRegularParameterized* rpi);

    //! Destructor
    virtual ~GateRegularParam();

    //! Implementation of virtual methods Construct and Destruct OwnPhysicalVolumes
    void ConstructOwnPhysicalVolume(G4bool flagUpdate);
    void DestroyGeometry();
    
    //! Get the parameterization
    inline GateRegularParameterization* GetParameterization() {return m_parameterization;}

private:

    GateRegularParameterized*          itsInserter;
    GateRegularParameterization*       m_parameterization;
    G4PVParameterised*                 m_pvParameterized;
};

#endif

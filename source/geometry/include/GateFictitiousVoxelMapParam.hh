/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#ifndef GATEFICTITIOUSVOXELMAPPARAM_HH
#define GATEFICTITIOUSVOXELMAPPARAM_HH 1

#include "globals.hh"
#include "GateBox.hh"
#include "GateRegularParameterization.hh"
#include "G4ThreeVector.hh"

class GateFictitiousVoxelMapParameterized;
class G4PVParameterised;

class GateFictitiousVoxelMapParam : public GateBox
{
public:

    //! Constructor
    GateFictitiousVoxelMapParam(const G4String& itsName, GateFictitiousVoxelMapParameterized* rpi);

    //! Destructor
    virtual ~GateFictitiousVoxelMapParam();

    //! Implementation of virtual methods Construct and Destruct OwnPhysicalVolumes
    void ConstructOwnPhysicalVolume(G4bool flagUpdate);
//    void DestroyOwnPhysicalVolumes();
    void DestroyGeometry();

    //! Get the parameterization
    inline GateRegularParameterization* GetParameterization() {return m_parameterization;}

private:

    GateFictitiousVoxelMapParameterized*  itsInserter;
    GateRegularParameterization*       m_parameterization;
    G4PVParameterised*                 m_pvParameterized;
};

#endif

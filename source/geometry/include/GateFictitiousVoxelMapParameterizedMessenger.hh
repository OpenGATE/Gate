/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#ifndef GateFictitiousVoxelMapParameterizedMessenger_hh
#define GateFictitiousVoxelMapParameterizedMessenger_hh 1

#include "globals.hh"
#include "GateMessenger.hh"

class GateFictitiousVoxelMapParameterized;

class GateFictitiousVoxelMapParameterizedMessenger : public GateMessenger
{
public:

    //! Constructor
    GateFictitiousVoxelMapParameterizedMessenger(GateFictitiousVoxelMapParameterized* itsInserter);

    //! Destructor
    ~GateFictitiousVoxelMapParameterizedMessenger();

    //! SetNewValue
    void SetNewValue(G4UIcommand*, G4String);

    //! Get the FictitiousVoxelMapParameterized
    virtual inline GateFictitiousVoxelMapParameterized* GetFictitiousVoxelMapParameterized()
      { return m_inserter; }

private:

    G4UIcmdWithoutParameter*        AttachPhantomSDCmd;
    G4UIcmdWithAString*             InsertReaderCmd;
    G4UIcmdWithoutParameter*        RemoveReaderCmd;
    G4UIcmdWithAString*             AddOutputCmd;
    G4UIcmdWithAnInteger*           VerboseCmd;
    G4UIcmdWithABool*               SkipEqualMaterialsCmd;
    G4UIcmdWithADoubleAndUnit*      FictitiousEnergyCmd;
    G4UIcmdWithADoubleAndUnit*      DiscardEnergyCmd;

    GateFictitiousVoxelMapParameterized*  m_inserter;
};

#endif

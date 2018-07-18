/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#ifndef GateDistributionMessenger_h
#define GateDistributionMessenger_h 1

#include "GateNamedObjectMessenger.hh"
#include <G4UIcommand.hh>

class G4UIdirectory;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWithoutParameter;
class GateVDistribution;

class GateDistributionMessenger: public GateNamedObjectMessenger
{
  public:
    GateDistributionMessenger(GateVDistribution* itsDistribution,
    			     const G4String& itsDirectoryName="");
    virtual ~GateDistributionMessenger();
    inline GateVDistribution* GetDistribution() const
           {return (GateVDistribution*)(GetNamedObject());}

    void SetNewValue(G4UIcommand* aCommand, G4String aString);
    void SetUnitX(const G4String& unitX);
    void SetUnitY(const G4String& unitY);
    inline G4String UnitCategoryX() const {return m_unitX.empty()?"":G4UIcommand::CategoryOf(m_unitX);}
    inline G4String UnitCategoryY() const {return m_unitY.empty()?"":G4UIcommand::CategoryOf(m_unitY);}

  private:
    G4String withUnity(G4double value,G4String category) const;
    G4UIcmdWithoutParameter        *getMinX_Cmd ;
    G4UIcmdWithoutParameter        *getMinY_Cmd ;
    G4UIcmdWithoutParameter        *getMaxX_Cmd ;
    G4UIcmdWithoutParameter        *getMaxY_Cmd ;
    G4UIcmdWithoutParameter        *getRandom_Cmd ;
    G4UIcmdWithADoubleAndUnit      *getValueCmd ;
    G4String	    	    	   m_unitX;
    G4String	    	    	   m_unitY;
};

#endif

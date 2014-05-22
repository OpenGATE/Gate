/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateDistributionListManagerMessenger_h
#define GateDistributionListManagerMessenger_h 1

#include <vector>
#include <map>

#include "GateListMessenger.hh"

class G4UIdirectory;
class G4UIcmdWithAString;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3Vector;
class G4UIcmdWith3VectorAndUnit;
class G4UIcmdWithoutParameter;
class G4UIcmdWithABool;
class G4UIcmdWithADouble;
class GateVDistribution;

class GateDistributionListManager;

class GateDistributionListMessenger: public GateListMessenger
{
  public:
    //! Constructor
    GateDistributionListMessenger(GateDistributionListManager* itsListManager);

    virtual ~GateDistributionListMessenger();  //!< destructor


    //! Lists all the Distribution-names into a string
    virtual const G4String& DumpMap();

    //! Virtual method: create and insert a new attachment
    virtual void DoInsertion(const G4String& typeName);

    //! Get the store pointer
    inline GateDistributionListManager* GetDistributionListManager()
      { return (GateDistributionListManager*) GetListManager(); }

  private:
   typedef enum {kFile,kManual,kGaussian,kExponential,kFlat} distType_t;
   static std::map<G4String,distType_t> fgkTypes;
   std::vector<GateVDistribution*> m_distribVector;
};

#endif

/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*!
  \class  GateSinglesDigitizer
   Ex-PulseProcessor : Last modification in 12/2011 by Abdul-Fattah.Mohamad-Hadi@subatech.in2p3.fr,
   for the multi-system approach.

    adapter for GND 2022 by olga.kochebina@cea.fr
*/

#ifndef GateSinglesDigitizer_h
#define GateSinglesDigitizer_h 1

#include "globals.hh"
#include <vector>

#include "GateModuleListManager.hh"
#include "GateDigi.hh"
#include "GateVDigitizerModule.hh"
#include "GateCrystalSD.hh"

class GateDigitizerMgr;
class G4VDigitizerModule;
class GateSinglesDigitizerMessenger;
class GatePulseList;
class GateVSystem;

class GateVDigitizerModule;

class GateSinglesDigitizer : public GateModuleListManager
{
  public:
    GateSinglesDigitizer(GateDigitizerMgr* itsDigitizerMgr,
    				const G4String& digitizerName,
    			    GateCrystalSD *SD);
    virtual ~GateSinglesDigitizer();


     const G4String& GetInputName() const
       { return m_inputName; }
     void SetInputName(const G4String& anInputName)
       {  m_inputName = anInputName; }

     inline GateCrystalSD* GetSD() const
            { return m_SD; }
     void SetSDname(GateCrystalSD* SD)
            {  m_SD = SD; }



     const G4String& GetOutputName() const
       { return m_outputName; }

     virtual inline GateVSystem* GetSystem() const
       { return m_system;}

     virtual inline void SetSystem(GateVSystem* aSystem)
       { m_system = aSystem; }



     virtual GateVDigitizerModule* GetDigitizerModule(size_t i)
           	  {return (GateVDigitizerModule*) GetElement(i);}

     GateVDigitizerModule* FindDigitizerModule(const G4String& name);


     void SetName(const G4String& anInputName)
        {  m_digitizerName = anInputName; }
     G4String GetName()
           {  return m_digitizerName; }


    void AddNewModule(GateVDigitizerModule* DM);

    void Describe(size_t indent=0);
    void DescribeMyself(size_t indent=0);

    G4String GetDMNameFromInsertionName(G4String name);

    //To set input collectionIDs for all DMs of this digitizer (m_DMlist)
    void SetDMCollectionIDs();
    //To set output collection ID for this digitizer, m_outputDigiCollectionID,
    //corresponding to the last DM output ID. Used by output modules to find what to write down
    void SetOutputCollectionID();



 protected:
      GateSinglesDigitizerMessenger*    m_messenger;
      GateVSystem *m_system;            //!< System to which the digitizer is attached
      G4String				   m_outputName;
      G4String                 m_inputName;

public:
      G4bool                m_recordFlag;

      std::vector<GateVDigitizerModule*>    	m_DMlist;	 //!< List of DigitizerModules for this digitizer
      GateCrystalSD*                m_SD;
      G4String                 m_digitizerName;

      G4int      m_outputDigiCollectionID;
};

#endif

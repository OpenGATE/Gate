/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*!
  \class  GateCoincidenceDigitizer
   Ex-PulseProcessor : Last modification in 12/2011 by Abdul-Fattah.Mohamad-Hadi@subatech.in2p3.fr,
   for the multi-system approach.

    adapter for GND 2022 by olga.kochebina@cea.fr
*/

#ifndef GateCoincidenceDigitizer_h
#define GateCoincidenceDigitizer_h 1

#include "globals.hh"
#include <vector>

#include "GateModuleListManager.hh"
#include "GateDigi.hh"
#include "GateVDigitizerModule.hh"
#include "GateCrystalSD.hh"

class GateDigitizerMgr;
class G4VDigitizerModule;
class GateCoincidenceDigitizerMessenger;
class GatePulseList;
class GateVSystem;

class GateVDigitizerModule;

class GateCoincidenceDigitizer : public GateModuleListManager
{
  public:
    GateCoincidenceDigitizer(GateDigitizerMgr* itsDigitizerMgr,
    				const G4String& digitizerName);
    virtual ~GateCoincidenceDigitizer();


    std::vector<G4String>& GetInputNames()
       { return m_inputNames; }
     void AddInputName(G4String anInputName)
       { m_inputNames.push_back(anInputName); }


     const G4String& GetOutputName() const
       { return m_outputName; }

     virtual inline GateVSystem* GetSystem() const
       { return m_system;}

     virtual inline void SetSystem(GateVSystem* aSystem)
       { m_system = aSystem; }



     virtual GateVDigitizerModule* GetDigitizerModule(size_t i)
           	  {return (GateVDigitizerModule*) GetElement(i);}

     GateVDigitizerModule* FindDigitizerModule(const G4String& name);


     void SetName(const G4String& aName)
        {  m_digitizerName = aName; }
     G4String GetName()
           {  return m_digitizerName; }


    void AddNewModule(GateVDigitizerModule* DM);

    void Describe(size_t indent=0);
    void DescribeMyself(size_t indent=0);

    G4String GetDMNameFromInsertionName(G4String name);

    // OK GND: obsolete
   // void SetNoPriority(G4bool b){m_noPriority = b;}
   // G4bool GetNoPriority(){return m_noPriority;}

    void SetCDMCollectionIDs();
    void SetOutputCollectionID();

 protected:
      GateCoincidenceDigitizerMessenger*    m_messenger;
      GateVSystem *m_system;            //!< System to which the digitizer is attached
      G4String				   m_outputName;
      std::vector<G4String>                m_inputNames;
      // OK GND: obsolete
      //G4bool         	      	           m_noPriority;



public:
      G4bool                m_recordFlag;

      std::vector<GateVDigitizerModule*>    	m_CDMlist;	 //!< List of DigitizerModules for this digitizer

      G4String                 m_digitizerName;
      G4int      m_outputDigiCollectionID;

};

#endif

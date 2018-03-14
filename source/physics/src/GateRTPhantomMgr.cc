/*----------------------
   OpenGATE Collaboration 
     
   Giovanni Santin <giovanni.santin@cern.ch> 
   Daniel Strul <daniel.strul@iphe.unil.ch> 
     
   Copyright (C) 2002,2003 UNIL/IPHE, CH-1015 Lausanne 

This software is distributed under the terms 
of the GNU Lesser General  Public Licence (LGPL) 
See LICENSE.md for further details 
----------------------*/

#include "GateRTPhantomMgr.hh"
#include "GateRTVPhantom.hh"



#include "GateOutputMgrMessenger.hh"
#include "GateRTPhantomMgrMessenger.hh"
#include "GateVSourceVoxelReader.hh"
#include "GateVGeometryVoxelReader.hh"
#include "GateVVolume.hh"

GateRTPhantomMgr* GateRTPhantomMgr::instance = 0;

void GateRTPhantomMgr::SetVerboseLevel(G4int val)
{ m_verboseLevel = val;

for (std::vector<GateRTPhantom*>::iterator itr = m_RTPhantom.begin(); itr != m_RTPhantom.end(); itr++)
(*itr)->SetVerboseLevel(val);

 }

GateRTPhantom * GateRTPhantomMgr::CheckSourceAttached( G4String aname)
{
GateRTPhantom * Ph = 0;

for (std::vector<GateRTPhantom*>::iterator itr = m_RTPhantom.begin(); itr != m_RTPhantom.end(); itr++)
{
if( (*itr)->GetSReader()->GetName() == aname )
    {Ph = (*itr);
     break;
    }
}
return Ph;      
}

GateRTPhantom * GateRTPhantomMgr::CheckGeometryAttached( G4String aname)
{
GateRTPhantom * Ph = 0;

for (std::vector<GateRTPhantom*>::iterator itr = m_RTPhantom.begin(); itr != m_RTPhantom.end(); itr++)
    {
     if( (*itr)->GetInserter()->GetObjectName() == aname )
     {Ph = *itr;
      break;
     }
    }
return Ph;      
}

GateRTPhantomMgr::GateRTPhantomMgr(const G4String name)
  : m_verboseLevel(2),
    m_messenger(0),
    m_name(name)
{
  m_messenger = new GateRTPhantomMgrMessenger(this);
}

GateRTPhantomMgr::~GateRTPhantomMgr()
{
  for (size_t iMod = 0; iMod < m_RTPhantom.size(); iMod++) {  // use iterator??
    delete m_RTPhantom[iMod];
  }
  m_RTPhantom.clear();
  delete m_messenger;
 if (m_verboseLevel > 0) G4cout << "GateRTPhantomMgr deleting...\n";


}

void GateRTPhantomMgr::UpdatePhantoms(G4double aTime)
{
 for (std::vector<GateRTPhantom*>::iterator itr = m_RTPhantom.begin(); itr != m_RTPhantom.end(); itr++)
   (*itr)->Compute(aTime);
}

void GateRTPhantomMgr::AddPhantom(G4String aname)
{
  if (m_verboseLevel > 2)
    G4cout << "GateRTPhantomMgr::AddRTPhantom\n";

  if ( aname == "RTVPhantom" )
  {
   GateRTVPhantom* aph = new GateRTVPhantom();
   aph->SetVoxellized();
   aph->Enable();
   m_RTPhantom.push_back(aph);
   return;
  }

  G4cout << "GateRTPhantomMgr::AddRTPhantom : ERROR : " << aname <<" is not recognized. IGNORED !!!"<< Gateendl;

}


void GateRTPhantomMgr::Describe()
{
  G4cout << "GateRTPhantomMgr name: " << m_name << Gateendl;
  G4cout << "Number of RTPhantoms inserted: " << m_RTPhantom.size() << Gateendl;
  G4cout << "Description of the single RTPhantoms: \n";

  for (std::vector<GateRTPhantom*>::iterator itr = m_RTPhantom.begin(); itr != m_RTPhantom.end(); itr++) {
    (*itr)->Describe();
    G4cout << "RTPhantom address : " << *itr << Gateendl;

  }
}

GateRTPhantom* GateRTPhantomMgr::Find( G4String aname)
{
GateRTPhantom* Ph = 0;
  for (std::vector<GateRTPhantom*>::iterator itr = m_RTPhantom.begin(); itr != m_RTPhantom.end(); itr++)
  {
   G4String cname = (*itr)->GetName();
   if (cname == aname) {
                        Ph = *itr;
                        break;    }
  }
return Ph;
}

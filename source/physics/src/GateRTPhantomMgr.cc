/*----------------------
   OpenGATE Collaboration 
     
   Giovanni Santin <giovanni.santin@cern.ch> 
   Daniel Strul <daniel.strul@iphe.unil.ch> 
     
   Copyright (C) 2002,2003 UNIL/IPHE, CH-1015 Lausanne 

This software is distributed under the terms 
of the GNU Lesser General  Public Licence (LGPL) 
See GATE/LICENSE.txt for further details 
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

for (size_t iMod = 0; iMod < m_RTPhantom.size(); iMod++)
m_RTPhantom[iMod]->SetVerboseLevel(val);

 }

GateRTPhantom * GateRTPhantomMgr::CheckSourceAttached( G4String aname)
{
GateRTPhantom * Ph = 0;

for (size_t iMod = 0; iMod < m_RTPhantom.size(); iMod++)
{
if( m_RTPhantom[iMod]->GetSReader()->GetName() == aname )
    {Ph = m_RTPhantom[iMod];
     break;
    }
}
return Ph;      
}

GateRTPhantom * GateRTPhantomMgr::CheckGeometryAttached( G4String aname)
{
GateRTPhantom * Ph = 0;

for (size_t iMod = 0; iMod < m_RTPhantom.size(); iMod++)
    {
     if( m_RTPhantom[iMod]->GetInserter()->GetObjectName() == aname )
     {Ph = m_RTPhantom[iMod];
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
  for (size_t iMod = 0; iMod < m_RTPhantom.size(); iMod++) {
    delete m_RTPhantom[iMod];
  }
  m_RTPhantom.clear();
  delete m_messenger;
 if (m_verboseLevel > 0) G4cout << "GateRTPhantomMgr deleting..." << G4endl;


}

void GateRTPhantomMgr::UpdatePhantoms(G4double aTime)
{
 for (size_t i = 0 ; i < m_RTPhantom.size(); i++)m_RTPhantom[i]->Compute(aTime);
}

void GateRTPhantomMgr::AddPhantom(G4String aname)
{
  if (m_verboseLevel > 2)
    G4cout << "GateRTPhantomMgr::AddRTPhantom" << G4endl;

  if ( aname == "RTVPhantom" )
  {
   GateRTVPhantom* aph = new GateRTVPhantom();
   aph->SetVoxellized();
   aph->Enable();
   m_RTPhantom.push_back(aph);
   return;
  }

  G4cout << "GateRTPhantomMgr::AddRTPhantom : ERROR : " << aname <<" is not recognized. IGNORED !!!"<< G4endl;

}


void GateRTPhantomMgr::Describe()
{
  G4cout << "GateRTPhantomMgr name: " << m_name << G4endl;
  G4cout << "Number of RTPhantoms inserted: " << m_RTPhantom.size() << G4endl;
  G4cout << "Description of the single RTPhantoms: " << G4endl;

  for (size_t iMod=0; iMod<m_RTPhantom.size(); iMod++) {
    m_RTPhantom[iMod]->Describe();
    G4cout << "RTPhantom address : " << m_RTPhantom[iMod] << G4endl;

  }
}

GateRTPhantom* GateRTPhantomMgr::Find( G4String aname)
{
GateRTPhantom* Ph = 0;
  for (size_t iMod=0; iMod<m_RTPhantom.size(); iMod++)
  {
   G4String cname = m_RTPhantom[iMod]->GetName();
   if (cname == aname) {
                        Ph = m_RTPhantom[iMod];
                        break;    }
  }
return Ph;
}

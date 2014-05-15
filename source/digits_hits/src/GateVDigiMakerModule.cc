/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateVDigiMakerModule.hh"

#include "G4DigiManager.hh"

#include "GateSingleDigi.hh"
#include "GateTools.hh"
#include "GateDigitizer.hh"

// Constructor
GateVDigiMakerModule::GateVDigiMakerModule( GateDigitizer* itsDigitizer,
      	      	         	            const G4String& itsInputName)
  :GateClockDependent(itsDigitizer->GetObjectName() + "/" + itsInputName + "/digiMaker",false),
   m_digitizer(itsDigitizer),
   m_inputName(itsInputName)
{
  G4String collectionName = itsInputName;

  if ( collectionName.substr(0,10) == "digitizer/" )
    collectionName = collectionName.substr(10);


  G4String::size_type pos;
  do {
    pos = collectionName.find_first_of('/');
    if (pos != G4String::npos) {
      collectionName.erase(pos,1);
      if ( pos < collectionName.length() )
        collectionName.at(pos) = toupper( collectionName.at(pos) ) ;
    }
  } while ( pos != G4String::npos);

  m_collectionName = collectionName;
}



// Destructor
GateVDigiMakerModule::~GateVDigiMakerModule()
{
}



void GateVDigiMakerModule::DescribeMyself(size_t indent)
{
  G4cout << GateTools::Indent(indent) << "Input pulses:       " << m_inputName << G4endl;
  G4cout << GateTools::Indent(indent) << "Output digis:       " << m_collectionName << G4endl;
}

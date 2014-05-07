/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateVOutputModule.hh"
//#include "GateOutputModuleMessenger.hh"
#include "GateTools.hh"

GateVOutputModule::GateVOutputModule(const G4String& name, GateOutputMgr* outputMgr,DigiMode digiMode)
  : m_outputMgr(outputMgr),
    m_name(name),
    m_digiMode(digiMode),
    m_isEnabled(false)
// !!!! By default all output modules will be disabled !!!!
// !!!! Think about it now for the derived classes !!!!
// !!!! So please do not enable by default the derived classes !!!!
// !!!! and think about using the mother variable members. !!!!
{
//    m_messenger = new GateOutputModuleMessenger(this);
}

GateVOutputModule::~GateVOutputModule()
{
//    delete m_messenger;
}

/* Virtual method to print-out a description of the module

	indent: the print-out indentation (cosmetic parameter)
*/
void GateVOutputModule::Describe(size_t indent)
{
  G4cout << G4endl << GateTools::Indent(indent) << "Output module: '" << m_name << "'" << G4endl;
}
